import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import mujoco
from collections import deque

tf.random.set_seed(42)
np.random.seed(42)


N=500
N_E=int(N*0.8)
N_I=N-N_E
EXC_IDS=list(range(0,400))
INH_IDS=list(range(400,500))

OBS_DIM=105
ACT_DIM=8

OBS_IN_IDS=list(range(0,105))
REW_IN_ID=[105]

NEUR_ACT=4
ACT_OUT_IDS=[]
for ai in range(ACT_DIM):
    st=106+ai*NEUR_ACT
    ACT_OUT_IDS.extend(range(st,st+NEUR_ACT))
OUT_IDS=ACT_OUT_IDS
FREE_N=list(range(138,N))

OBS_SCALE=2.0
REW_SCALE=0.5
ACT_SCALE=1.0
ACT_WIN=15
ACT_DEC_MODE="rate"

V_TH_EXC=0.9
V_TH_INH=0.3
V_RST=0.0
V_CLAMP=-2.0
LR_LTP=0.0009
LR_LTD=0.0008
ELIG_DECAY=0.9
W_MAX=5.0
W_MIN_EXC=0.0
W_MIN_INH=-8.0
BASE_LR=0.005
INIT_BASE=0.0
DOPA_CLIP=5.0

TARGET_FR=0.10
SYNAP_SCALE_INTERVAL=100
ADAPT_RISE=0.15
ADAPT_DECAY=0.90
WD_RATE=0.9999

NEURO_INTERVAL=1000
N_REPLACE=5
ACT_THRESH=0.02
EXCLUDE_REPL=(OBS_IN_IDS+REW_IN_ID+ACT_OUT_IDS)

INNER_STEPS=30


# e/i constraints
def enforce_ei_constraints(W):
    W=tf.cast(W,tf.float32)
    exc_mask=tf.constant([i in EXC_IDS for i in range(N)],dtype=tf.bool)
    inh_mask=tf.logical_not(exc_mask)
    W=tf.where(exc_mask[None,:],tf.maximum(W,0.0),W)
    W=tf.where(inh_mask[None,:],tf.minimum(W,0.0),W)
    W=tf.where(inh_mask[None,:],W*0.85,W)
    W=tf.clip_by_value(W,W_MIN_INH,W_MAX)
    W=tf.linalg.set_diag(W,tf.zeros(N))
    return W


class Net:
    def __init__(self):
        self.n=N
        self.is_exc=tf.constant([1.0 if i in EXC_IDS else 0.0 for i in range(N)],dtype=tf.float32)
        self.is_inh=tf.constant([1.0 if i in INH_IDS else 0.0 for i in range(N)],dtype=tf.float32)

        self.base_thr=tf.constant([V_TH_EXC if i in EXC_IDS else V_TH_INH for i in range(N)],dtype=tf.float32)

        W_raw=tf.random.uniform((N,N),-0.002,0.002,dtype=tf.float32)
        W_init=enforce_ei_constraints(W_raw)
        self.W=tf.Variable(W_init,dtype=tf.float32,name="W")

        self.V=tf.Variable(tf.zeros(N),dtype=tf.float32,name="V")
        self.adapt_thr=tf.Variable(self.base_thr,dtype=tf.float32,name="adapt_thr")
        self.spikes=tf.Variable(tf.zeros(N),dtype=tf.float32)
        self.prev_spikes=tf.Variable(tf.zeros(N),dtype=tf.float32)
        self.elig=tf.Variable(tf.zeros((N,N)),dtype=tf.float32,name="elig")

        bias_init=tf.concat([tf.random.uniform((N_E,),0.001,0.005),tf.random.uniform((N_I,),-0.35,-0.25)],axis=0)
        out_idx=tf.constant(OUT_IDS,dtype=tf.int32)
        out_bias=tf.random.uniform((len(OUT_IDS),),0.5,0.6)
        bias_init=tf.tensor_scatter_nd_update(bias_init,tf.expand_dims(out_idx,1),out_bias)
        self.bias=tf.Variable(bias_init,dtype=tf.float32,name="bias")

        self.baseline=tf.Variable(INIT_BASE,dtype=tf.float32)

        self.scale_spike_cnt=tf.Variable(tf.zeros(N),dtype=tf.float32)
        self.neuro_spike_cnt=tf.Variable(tf.zeros(N),dtype=tf.float32)
        self.scale_steps=tf.Variable(0,dtype=tf.int32)
        self.neuro_steps=tf.Variable(0,dtype=tf.int32)

        self.replaced_count=tf.Variable(0,dtype=tf.int32)
        self.repl_history=[]
        self.spike_buf=deque(maxlen=ACT_WIN)
        self.total_rew=tf.Variable(0.0,dtype=tf.float32)
        self.total_dopa=tf.Variable(0.0,dtype=tf.float32)

    def reset(self):
        self.V.assign(tf.zeros(N))
        self.adapt_thr.assign(self.base_thr)
        self.spikes.assign(tf.zeros(N))
        self.prev_spikes.assign(tf.zeros(N))
        self.elig.assign(tf.zeros((N,N)))
        self.scale_spike_cnt.assign(tf.zeros(N))
        self.neuro_spike_cnt.assign(tf.zeros(N))
        self.scale_steps.assign(0)
        self.neuro_steps.assign(0)
        self.spike_buf.clear()
        self.total_rew.assign(0.0)
        self.total_dopa.assign(0.0)

    def constrain(self):
        self.W.assign(enforce_ei_constraints(self.W))


@tf.function
def external_input(obs, rew_val):
    ext=tf.zeros(N,dtype=tf.float32)
    obs=tf.cast(obs,tf.float32)
    obs=tf.clip_by_value(obs,-10.0,10.0)*OBS_SCALE
    idx=tf.constant(OBS_IN_IDS,dtype=tf.int32)
    ext=tf.tensor_scatter_nd_update(ext,tf.expand_dims(idx,1),obs)
    rew=tf.cast(rew_val,tf.float32)
    rew=tf.clip_by_value(rew,-10.0,10.0)*REW_SCALE
    ridx=tf.constant(REW_IN_ID,dtype=tf.int32)
    ext=tf.tensor_scatter_nd_update(ext,tf.expand_dims(ridx,1),tf.ones(1)*rew)
    return ext


@tf.function
def upd_neurons(V,W,bias,prev_spikes,ext_input,adapt_thr,base_thr):
    wsum=tf.linalg.matvec(W,prev_spikes)
    V_new=0.95*V+wsum+bias+ext_input
    V_new=tf.maximum(V_new,V_CLAMP)
    spikes_new=tf.cast(V_new>=adapt_thr,tf.float32)
    V_new=tf.where(spikes_new>0,V_RST,V_new)
    thr_new=base_thr+(adapt_thr-base_thr)*ADAPT_DECAY+spikes_new*ADAPT_RISE
    return V_new,spikes_new,thr_new


@tf.function
def mod_weights(W,elig,dopa):
    dopa=tf.cast(dopa,tf.float32)
    ltp=LR_LTP*tf.maximum(dopa,0.0)*elig
    ltd=LR_LTD*tf.minimum(dopa,0.0)*elig
    W_new=W+ltp+ltd
    exc_mask=tf.constant([i in EXC_IDS for i in range(N)],dtype=tf.bool)
    inh_mask=tf.logical_not(exc_mask)
    W_new=tf.where(exc_mask[None,:],tf.maximum(W_new,0.0),W_new)
    W_new=tf.where(inh_mask[None,:],tf.minimum(W_new,0.0),W_new)
    W_new=tf.clip_by_value(W_new,W_MIN_INH,W_MAX)
    W_new=tf.linalg.set_diag(W_new,tf.zeros(N))
    return W_new,ltp+ltd


@tf.function
def syn_scale(W,spike_cnt,total_steps,target_rate=TARGET_FR):
    total_steps=tf.cast(tf.maximum(total_steps,1),tf.float32)
    actual_rate=spike_cnt/total_steps
    scaling=target_rate/(actual_rate+1e-6)
    scaling=tf.clip_by_value(scaling,0.5,2.0)
    W_scaled=W*tf.reshape(scaling,(-1,1))
    W_scaled=tf.linalg.set_diag(W_scaled,tf.zeros(N))
    return W_scaled


@tf.function
def w_decay(W,rate=WD_RATE):
    W_decayed=W*rate
    W_decayed=tf.linalg.set_diag(W_decayed,tf.zeros(N))
    return W_decayed


def decode(net):
    out_spikes=tf.gather(net.spikes,OUT_IDS)
    out_spikes=tf.reshape(out_spikes,(ACT_DIM,NEUR_ACT))
    rates=tf.reduce_mean(out_spikes,axis=1)
    action=rates*2.0-1.0
    action*=ACT_SCALE
    action=tf.clip_by_value(action,-1.0,1.0)
    return action.numpy().astype(np.float32)


def neurogen(net):
    total_steps=tf.cast(tf.maximum(net.neuro_steps,1),tf.float32)
    firing_rates=net.neuro_spike_cnt/total_steps
    protected_mask=tf.constant([1.0 if i in EXCLUDE_REPL else 0.0 for i in range(N)],dtype=tf.float32)
    below=tf.cast(firing_rates<ACT_THRESH,tf.float32)
    candidates=below*(1.0-protected_mask)
    cand_idx=tf.where(candidates>0)[:,0]
    all_non=tf.where(protected_mask==0)[:,0]

    if int(cand_idx.shape[0])>=N_REPLACE:
        cand_rates=tf.gather(firing_rates,cand_idx)
        sorted_idx=tf.argsort(cand_rates)
        to_replace=tf.gather(cand_idx,sorted_idx[:N_REPLACE])
    else:
        all_rates=tf.gather(firing_rates,all_non)
        sorted_idx=tf.argsort(all_rates)
        to_replace=tf.gather(all_non,sorted_idx[:N_REPLACE])

    W_np=net.W.numpy()
    V_np=net.V.numpy()
    thr_np=net.adapt_thr.numpy()
    replaced=[]

    for nid_t in to_replace:
        nid=int(nid_t.numpy())
        replaced.append(nid)
        if nid in EXC_IDS:
            W_np[:,nid]=np.random.uniform(0.01,0.10,N)
        else:
            W_np[:,nid]=-np.random.uniform(0.01,0.10,N)
        for src in range(N):
            if src==nid: continue
            if src in EXC_IDS:
                W_np[nid,src]=np.random.uniform(0.01,0.10)
            else:
                W_np[nid,src]=-np.random.uniform(0.01,0.10)
        W_np[nid,nid]=0.0
        V_np[nid]=0.0
        thr_np[nid]=net.base_thr.numpy()[nid]
        net.elig[nid,:].assign(tf.zeros(N))
        net.elig[:,nid].assign(tf.zeros(N))
        net.spikes[nid].assign(0.0)
        net.prev_spikes[nid].assign(0.0)

    net.W.assign(tf.constant(W_np,dtype=tf.float32))
    net.V.assign(tf.constant(V_np,dtype=tf.float32))
    net.adapt_thr.assign(tf.constant(thr_np,dtype=tf.float32))
    net.constrain()
    net.replaced_count.assign_add(len(replaced))
    net.repl_history.append(replaced)
    print("-------------------------------------")
    print(replaced)
    print('-----------------------------------------')
    return replaced


def run_step(net,obs,rew_val,step_cnt=0):
    ext=external_input(obs,rew_val)
    for _ in range(INNER_STEPS):
        V_new,spikes_new,thr_new=upd_neurons(
            net.V,net.W,net.bias,net.prev_spikes,ext,
            net.adapt_thr,net.base_thr
        )
        pre=net.prev_spikes
        post=spikes_new
        spike_pair=post[:,None]*pre[None,:]
        net.elig.assign(ELIG_DECAY*net.elig+spike_pair)
        net.scale_spike_cnt.assign_add(spikes_new)
        net.neuro_spike_cnt.assign_add(spikes_new)
        net.scale_steps.assign_add(1)
        net.neuro_steps.assign_add(1)
        net.V.assign(V_new)
        net.spikes.assign(spikes_new)
        net.prev_spikes.assign(spikes_new)
        net.adapt_thr.assign(thr_new)

    rew_t=tf.cast(rew_val,tf.float32)
    dopa=rew_t-net.baseline
    dopa=tf.clip_by_value(dopa,-DOPA_CLIP,DOPA_CLIP)
    net.baseline.assign(net.baseline+BASE_LR*dopa)
    net.total_rew.assign_add(rew_t)
    net.total_dopa.assign_add(dopa)

    W_new,dw=mod_weights(net.W,net.elig,dopa)
    net.W.assign(W_new)
    net.W.assign(w_decay(net.W))
    net.constrain()

    mean_dw=float(tf.reduce_mean(tf.abs(dw)).numpy())
    max_dw=float(tf.reduce_max(tf.abs(dw)).numpy())
    mean_elig=float(tf.reduce_mean(tf.abs(net.elig)).numpy())

    if int(net.scale_steps.numpy())>=SYNAP_SCALE_INTERVAL:
        net.W.assign(syn_scale(net.W,net.scale_spike_cnt,net.scale_steps))
        net.constrain()
        net.scale_spike_cnt.assign(tf.zeros(N))
        net.scale_steps.assign(0)

    replaced=[]
    if int(net.neuro_steps.numpy())>=NEURO_INTERVAL:
        replaced=neurogen(net)
        net.neuro_spike_cnt.assign(tf.zeros(N))
        net.neuro_steps.assign(0)

    action=decode(net)

    if int(net.scale_steps.numpy())>0:
        avg_fr=float(tf.reduce_mean(net.scale_spike_cnt/tf.cast(net.scale_steps,tf.float32)).numpy())
    else:
        avg_fr=0.0
    active=int(tf.reduce_sum(net.spikes).numpy())

    return action, {
        "dopa":float(dopa.numpy()),
        "base":float(net.baseline.numpy()),
        "avg_fr":avg_fr,
        "active":active,
        "dw":mean_dw,
        "max_dw":max_dw,
        "elig":mean_elig,
        "replaced":replaced,
    }


def add_noise(actions,explore=0.05):
    noise=np.random.normal(0,explore,size=actions.shape)
    noisy=actions+noise
    return np.clip(noisy,-1.0,1.0)


def plot_progress(ep,rewards,baselines,dopas,w_changes,eligibilities,frs,neuro_events=None,save_path=None):
    fig, axes=plt.subplots(2,3,figsize=(15,8))
    ax1,ax2,ax3,ax4,ax5,ax6=axes.flatten()
    window=10
    if len(rewards)>=window:
        ma=np.convolve(rewards,np.ones(window)/window,mode='valid')
        ax1.plot(range(window-1,len(rewards)),ma,'r-',label='Mov avg (10)')
        ax1.legend()
    ax1.plot(rewards,'b-')
    ax1.set_title('Total Reward per Episode')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True)

    ax2.plot(baselines,'g-')
    ax2.set_title('Dopamine Baseline')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Baseline')
    ax2.grid(True)

    ax3.plot(dopas,'r-')
    ax3.set_title('Mean Dopamine (TD err)')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Dopamine')
    ax3.axhline(0,color='k',linestyle='--',alpha=0.5)
    ax3.grid(True)

    ax4.plot(frs,'m-')
    ax4.set_title('Mean Firing Rate')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Firing rate')
    ax4.axhline(TARGET_FR,color='k',linestyle='--',alpha=0.5,label='Target')
    ax4.legend()
    ax4.grid(True)

    ax5.plot(w_changes,'c-')
    ax5.set_title('Mean |ΔW|')
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('|ΔW|')
    ax5.grid(True)

    ax6.plot(eligibilities,'orange')
    ax6.set_title('Mean Eligibility')
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('|E|')
    ax6.grid(True)

    if neuro_events:
        for epi,st,neurs in neuro_events:
            if epi<=ep:
                ax1.axvline(x=epi,color='r',alpha=0.15,linestyle='--',linewidth=0.8)
    plt.suptitle(f'Trainig Progress – Episode {ep}',fontsize=14)
    plt.tight_layout(rect=[0,0,1,0.95])
    if save_path:
        plt.savefig(save_path,dpi=150)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    plt.close(fig)


env=gym.make("Ant-v5",render_mode="human",ctrl_cost_weight=0.0,healthy_z_range=(0.5,1.0),healthy_reward=1)

net=Net()
print()
print("net init!")
print("W shape:",net.W.shape)
print("init W mean:",float(tf.reduce_mean(tf.abs(net.W)).numpy()))
print("init baseline:",net.baseline.numpy())

N_EPISODES=200
MAX_STEPS=1000

ep_rewards=[]
ep_lengths=[]
all_actions=[]
base_hist=[]
dopa_hist=[]
dw_hist=[]
elig_hist=[]
fr_hist=[]
neuro_events=[]

print()
print("Starting Ant-v5 train...")
print(f"Episodes: {N_EPISODES}")
print(f"Max steps: {MAX_STEPS}")
print(f"Inner SNN steps: {INNER_STEPS}")
print()

for ep in range(N_EPISODES):
    obs,info=env.reset()
    net.reset()
    total_rew=0.0
    ep_acts=[]
    ep_dopa=[]
    ep_dw=[]
    ep_elig=[]
    ep_fr=[]
    rew=0.0
    for st in range(MAX_STEPS):
        acts,info_dict=run_step(net,obs,rew,st)
        if ep<30:
            if np.random.random()<0.7:
                acts=np.random.uniform(-1.0,1.0,ACT_DIM)
            else:
                acts=add_noise(acts,0.5)
        elif ep<50:
            acts=add_noise(acts,0.3)
        else:
            acts=add_noise(acts,0.1)
        obs,rew,terminated,truncated,info=env.step(acts)
        total_rew+=rew
        ep_acts.append(acts)
        ep_dopa.append(info_dict["dopa"])
        ep_dw.append(info_dict["dw"])
        ep_elig.append(info_dict["elig"])
        ep_fr.append(info_dict["avg_fr"])
        print("V:",float(tf.reduce_mean(net.V).numpy()),
              "spikes:",int(tf.reduce_sum(net.spikes).numpy()),
              "elig:",float(tf.reduce_mean(tf.abs(net.elig)).numpy()))
        if info_dict["replaced"]:
            neuro_events.append((ep,st,info_dict["replaced"]))
        if st%100==0:
            print(f"[Ep {ep:03d} | Step {st:04d}] Rew={rew:.4f} | D={info_dict['dopa']:.4f} | |dW|={info_dict['dw']:.8f} | E={info_dict['elig']:.6f} | FR={info_dict['avg_fr']:.4f}")
        if terminated or truncated:
            break

    ep_rewards.append(total_rew)
    ep_lengths.append(st+1)
    all_actions.append(np.array(ep_acts))
    base_hist.append(net.baseline.numpy())
    dopa_hist.append(np.mean(ep_dopa))
    dw_hist.append(np.mean(ep_dw))
    elig_hist.append(np.mean(ep_elig))
    fr_hist.append(np.mean(ep_fr))

    if ep%10==0:
        avg_rew=np.mean(ep_rewards[-10:])
        curr_W_mean=float(tf.reduce_mean(tf.abs(net.W)).numpy())
        plot_progress(ep,ep_rewards,base_hist,dopa_hist,dw_hist,elig_hist,fr_hist,neuro_events,save_path=f"progress_ep{ep:03d}.png")
        W_final=net.W.numpy()
        vals=W_final[np.abs(W_final)>0]
        max_abs=np.percentile(np.abs(vals),95)
        if max_abs<1e-6:
            max_abs=1.0
        plt.figure(figsize=(8,6))
        plt.imshow(W_final,cmap='RdBu_r',aspect='auto',vmin=-max_abs,vmax=max_abs)
        plt.colorbar(label='Synaptic weight')
        plt.title('Final Weight Matrix (scaled by 95th pctl)')
        plt.xlabel('Presynaptic')
        plt.ylabel('Postsynaptic')
        plt.savefig('weight_matrix_final_scaled.png',dpi=150)

env.close()