import os,json
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
import numpy as np
import gymnasium as gym
import gymnasium_robotics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import TQC

gym.register_envs(gymnasium_robotics)

ALGO         = 'TQC_HER_6M'
TOTAL_STEPS  = 6_000_000
SEEDS        = [0, 1, 2]
LOG_INTERVAL = 10_000
SAVE_FREQ    = 100_000
ENV_ID       = 'HandManipulateBlockRotateXYZ-v1'
RUN_DIR      = f'results/{ALGO}'

os.makedirs(RUN_DIR, exist_ok=True)

HER_KWARGS = dict(
    n_sampled_goal=4,
    goal_selection_strategy='future')

BASE_KWARGS = dict(
    policy='MultiInputPolicy',
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=HER_KWARGS,
    learning_rate=1e-3,
    buffer_size=1_000_000,
    learning_starts=1000,
    batch_size=256,
    tau=0.05,
    gamma=0.95,
    train_freq=1,
    gradient_steps=1,
    policy_kwargs=dict(net_arch=[256,256]),
    tensorboard_log=f'{RUN_DIR}/tb',
    device='cuda',
    verbose=0)

class CB(BaseCallback):
    def __init__(self,seed):
        super().__init__()
        self.seed=seed
        self.b=[]
        self.st=[]
        self.sr=[]
    def _on_step(self):
        for i in self.locals.get('infos',[]):
            if 'is_success' in i:
                self.b.append(float(i['is_success']))
        if self.num_timesteps%LOG_INTERVAL==0 and self.b:
            m=np.mean(self.b[-100:])
            self.st.append(self.num_timesteps)
            self.sr.append(m)
            print(f'[{ALGO}|s{self.seed}] '
                  f'step={self.num_timesteps} '
                  f'sr={m:.4f}',flush=True)
        return True
    def save(self):
        json.dump(
            {'algo':ALGO,'seed':self.seed,
             'steps':self.st,'success_rate':self.sr},
            open(f'{RUN_DIR}/seed_{self.seed}.json','w'),
            indent=2)
        print(f'✅ Saved seed_{self.seed}.json')

def make_env():
    env=gym.make(ENV_ID,reward_type='sparse')
    env=Monitor(env)
    return env

all_s,all_t=[],None
for seed in SEEDS:
    print(f'\n=== {ALGO} SEED {seed} 6M ===')
    env=DummyVecEnv([make_env])
    cb=CB(seed)
    checkpoint_cb=CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=RUN_DIR,
        name_prefix=f'model_seed_{seed}',
        save_replay_buffer=False,
        save_vecnormalize=False)
    model=TQC(env=env,seed=seed,**BASE_KWARGS)
    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=[cb,checkpoint_cb],
        reset_num_timesteps=True)
    cb.save()
    model.save(f'{RUN_DIR}/model_seed_{seed}')
    env.close()
    all_s.append(cb.sr)
    all_t=cb.st
    print(f'✅ Seed {seed} DONE')

sns.set_style('darkgrid')
SC=['#aec7e8','#ffbb78','#98df8a']
n=min(len(s) for s in all_s)
if n == 0:
    raise RuntimeError('No success-rate points were logged; check callback/env infos before plotting.')
arr=np.array([s[:n] for s in all_s])
mean=arr.mean(0);std=arr.std(0)
st=np.array(all_t[:n])
fig,ax=plt.subplots(figsize=(10,6))
for i in range(3):
    ax.plot(st,all_s[i][:n],alpha=0.4,
            color=SC[i],linewidth=1,
            label=f'Seed {i}')
ax.plot(st,mean,color='darkgreen',
        linewidth=2.5,label='Mean')
ax.fill_between(st,mean-std,mean+std,
                alpha=0.2,color='darkgreen')
ax.set_title('TQC+HER | 6M Steps | 3 Seeds\n'
             'HandManipulate XYZ | Sparse Reward')
ax.set_xlabel('Timesteps')
ax.set_ylabel('Success Rate')
ax.set_ylim(0,1)
ax.legend()
plt.tight_layout()
plt.savefig(f'{RUN_DIR}/{ALGO}_curve.png',dpi=150)
plt.show()
print(' ALL DONE: TQC_HER_6M')
