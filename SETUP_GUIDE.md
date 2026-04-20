# Full Setup & Run Guide for A5000 Remote PC

Complete step-by-step instructions to clone, install, and run the RL training on your A5000 GPU machine.

## Prerequisites on Remote Machine

Ensure the A5000 machine has:
- **Linux or Windows OS** (instructions below are for both)
- **Git** installed
- **Conda** (Miniconda or Anaconda)
- **NVIDIA Driver** compatible with CUDA 11.8 (verify with `nvidia-smi`)
- **SSH access** (if connecting remotely)

---

## Step-by-Step Setup

### Step 1: SSH into Remote Machine (if needed)

**From your local machine:**

```bash
ssh user@remote_ip
# or
ssh user@remote_hostname
```

Example:
```bash
ssh pujan@192.168.1.100
```

### Step 2: Clone Repository

```bash
git clone https://github.com/PujanPurohit/RL_Project.git
cd RL_Project
```

### Step 3: Create Conda Environment

```bash
conda create -n rl_project python=3.11 -y
conda activate rl_project
```

### Step 4: Install Core Dependencies

```bash
pip install stable-baselines3 gymnasium gymnasium-robotics numpy
```

### Step 5: Install CUDA-enabled PyTorch

For A5000 (NVIDIA CUDA capable):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

This installs PyTorch 2.7.1 with CUDA 11.8 support.

### Step 6: Install Logging & Progress Dependencies

```bash
pip install tensorboard tqdm rich
```

### Step 7: Verify GPU Setup

Check if GPU is detected:

```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

**Expected output:**
```
CUDA Available: True
GPU: NVIDIA A5000
```

Also verify with NVIDIA tools:
```bash
nvidia-smi
```

### Step 8: Test Environment (Optional but Recommended)

```bash
python test_env.py
```

Should output:
```
Starting test script...
Importing config...
ENV_ID: HandManipulateBlockRotateXYZ-v1
...
Test successful!
```

---

## Running Training

### Start Training

```bash
python train.py
```

You should see:
```
Script started!

==================================================
  Training : SAC + sparse reward
  Run name : SAC_sparse_1776710643
==================================================

Using cuda device
Wrapping the env with a `Monitor` wrapper
Wrapping the env in a DummyVecEnv.
Starting training ... (Ctrl+C to stop early)

Logging to outputs/logs/SAC_sparse_1776710643/SAC_1
```

### Run in Background (Recommended for SSH)

If you're SSH'd in and want to close the terminal without stopping training:

#### Option A: Using `screen` (easiest)

```bash
screen -S rl_training
conda activate rl_project
python train.py
# Press Ctrl+A then D to detach
```

Reattach later:
```bash
screen -r rl_training
```

#### Option B: Using `nohup`

```bash
conda activate rl_project
nohup python train.py > training.log 2>&1 &
```

Watch the log:
```bash
tail -f training.log
```

#### Option C: Using `tmux` (more powerful)

```bash
tmux new-session -d -s rl_training
tmux send-keys -t rl_training "conda activate rl_project && python train.py" Enter
```

List sessions:
```bash
tmux ls
```

Attach:
```bash
tmux attach -t rl_training
```

---

## Monitoring Training

### Monitor in Real-time (While SSH'd in)

```bash
tail -f training.log
```

### View TensorBoard on Remote Machine

In a separate terminal:

```bash
conda activate rl_project
tensorboard --logdir outputs/logs
```

#### View TensorBoard on Local Machine

**Option 1: SSH Port Forwarding**

On your local machine:
```bash
ssh -L 6006:localhost:6006 user@remote_ip
```

Then open: http://localhost:6006

**Option 2: Copy logs to local via SCP**

```bash
scp -r user@remote_ip:/path/to/RL_Project/outputs/logs ./local_logs
tensorboard --logdir local_logs
```

---

## Configuration Before Training

Edit `config.py` to change training parameters:

```bash
nano config.py
# or
vim config.py
```

Key settings:
- `ALGORITHM`: "SAC" or "PPO"
- `REWARD_TYPE`: "sparse" (good for complex tasks) or "dense"
- `TOTAL_STEPS`: Training duration (500,000 = ~hours depending on GPU)
- `SAVE_FREQ`: Checkpoint save frequency

Example configuration for faster testing:
```python
TOTAL_STEPS = 50_000  # Quick test run
ALGORITHM = "SAC"
REWARD_TYPE = "sparse"
```

---

## Output Structure

After training starts, check outputs:

```bash
ls -la outputs/
```

Directory structure:
```
outputs/
├── models/
│   └── SAC_sparse_1776710643/
│       ├── final_model.zip              # Final trained model
│       ├── best_model.zip               # Best checkpoint
│       └── smart_hand_model_*_steps.zip # Training checkpoints
└── logs/
    └── SAC_sparse_1776710643/           # TensorBoard logs
        └── SAC_1/
            └── events.out.tfevents.*
```

---

## Expected Performance on A5000

- **Training time** for 500k steps: ~2-4 hours (A5000 is very fast for RL)
- **Memory usage**: ~8-12GB VRAM (monitor with `nvidia-smi`)
- **Training throughput**: ~5,000-10,000 steps/min

Check GPU usage while training:
```bash
nvidia-smi -l 1  # Updates every 1 second
```

---

## Stopping Training

### If Training is Running Locally

Press `Ctrl+C` to gracefully stop (will save final model)

### If Running in Screen

```bash
screen -S rl_training
# Inside screen: Ctrl+C
# Detach: Ctrl+A then D
```

### If Running with nohup

```bash
pkill -f "python train.py"
```

---

## Troubleshooting

### Problem: "CUDA Available: False"

**Solution:**
1. Check NVIDIA driver: `nvidia-smi`
2. Reinstall PyTorch with CUDA:
   ```bash
   pip uninstall torch -y
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### Problem: "ImportError: gymnasium-robotics"

**Solution:**
```bash
pip install gymnasium-robotics
```

### Problem: "No module named stable_baselines3"

**Solution:**
```bash
pip install stable-baselines3
```

### Problem: GPU runs out of memory

**Solution** (in `train.py`, reduce batch size):
```python
SAC_CONFIG = dict(
    batch_size = 128,  # Reduce from 256
    ...
)
```

### Problem: Training is very slow

**Check:**
1. Verify GPU is being used: `nvidia-smi` should show GPU usage
2. Check if CPU is maxed out: `top` command
3. Check disk space: `df -h`

### Problem: Can't SSH into machine

**Debug:**
```bash
ping remote_ip
ssh -v user@remote_ip  # Verbose mode
```

---

## After Training

### Download Trained Model

From your local machine:

```bash
scp -r user@remote_ip:/path/to/RL_Project/outputs/models/SAC_sparse_* ./local_models/
```

### Use Trained Model

Load and test the model:

```python
from stable_baselines3 import SAC
model = SAC.load("outputs/models/SAC_sparse_1776710643/best_model")

from env_wrapper import make_env
env = make_env(reward_type="sparse")
obs, info = env.reset()

for _ in range(100):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```

---

## Quick Command Reference

```bash
# Activate environment
conda activate rl_project

# Clone repo
git clone https://github.com/PujanPurohit/RL_Project.git

# Install all deps (one-liner)
pip install stable-baselines3 gymnasium gymnasium-robotics numpy tensorboard tqdm rich torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Test GPU
python -c "import torch; print(torch.cuda.is_available())"

# Start training (foreground)
python train.py

# Start training (background with screen)
screen -S rl_training
python train.py
# Ctrl+A then D to detach

# Monitor TensorBoard
tensorboard --logdir outputs/logs

# View training logs
tail -f training.log

# Check GPU usage
nvidia-smi -l 1

# Stop training
Ctrl+C
```

---

## Need Help?

If you encounter issues:

1. Check the main [README.md](README.md)
2. Review error messages carefully (they usually indicate what's missing)
3. Verify all steps were completed in order
4. For CUDA issues, check NVIDIA documentation for your driver version
5. For environment issues, ensure `conda activate rl_project` is run before each command

Good luck with your training! The A5000 should complete training much faster than CPU. 🚀
