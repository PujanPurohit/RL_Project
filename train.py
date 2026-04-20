# ─── train.py ──────────────────────────────────────────────
import os
import sys
import time
from config import (ALGORITHM, REWARD_TYPE, TOTAL_STEPS,
                    SAVE_FREQ, PPO_CONFIG, SAC_CONFIG, HER_CONFIG)
from env_wrapper import make_env

from stable_baselines3 import PPO, SAC
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

print("Script started!", flush=True)

run_name  = f"{ALGORITHM}_{REWARD_TYPE}_{int(time.time())}"
model_dir = os.path.join("outputs", "models", run_name)
log_dir   = os.path.join("outputs", "logs",   run_name)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir,   exist_ok=True)

print(f"\n{'='*50}", flush=True)
print(f"  Training : {ALGORITHM} + {REWARD_TYPE} reward", flush=True)
print(f"  Run name : {run_name}", flush=True)
print(f"{'='*50}\n", flush=True)

env      = make_env(reward_type=REWARD_TYPE)
eval_env = make_env(reward_type=REWARD_TYPE)


if ALGORITHM == "PPO":
    model = PPO(
        "MultiInputPolicy",
        env,
        tensorboard_log=log_dir,
        device="cuda",
        **PPO_CONFIG,
    )

elif ALGORITHM == "SAC":
    if REWARD_TYPE == "sparse":
        model = SAC(
            "MultiInputPolicy",
            env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=HER_CONFIG,
            tensorboard_log=log_dir,
            device="cuda",
            **SAC_CONFIG,
        )
    else:
        model = SAC(
            "MultiInputPolicy",
            env,
            tensorboard_log=log_dir,
            device="cuda",
            **SAC_CONFIG,
        )


checkpoint_cb = CheckpointCallback(
    save_freq=SAVE_FREQ,
    save_path=model_dir,
    name_prefix="smart_hand_model",
)

eval_cb = EvalCallback(
    eval_env,
    best_model_save_path=model_dir,
    log_path=log_dir,
    eval_freq=10_000,
    n_eval_episodes=20,
    deterministic=True,
    verbose=1,
)

# ── Train ──────────────────────────────────────────────────
print("Starting training ... (Ctrl+C to stop early)\n")
model.learn(
    total_timesteps=TOTAL_STEPS,
    callback=[checkpoint_cb, eval_cb],
    progress_bar=True,
)

# ── Save final model ───────────────────────────────────────
final_path = os.path.join(model_dir, "final_model")
model.save(final_path)
print(f"\nFinal model → {final_path}.zip")
print(f"Best model  → {model_dir}/best_model.zip")
print(f"RUN_NAME: {run_name}")
env.close()
eval_env.close()