# RL Project - SAC/PPO Training with Gymnasium Robotics

Reinforcement Learning training using Stable Baselines3 with SAC or PPO algorithms on the Hand Manipulation environment.

## Prerequisites

- Python 3.11
- Conda (Miniconda or Anaconda)
- CUDA 11.8 (for GPU support on A5000 or similar NVIDIA GPUs)
- NVIDIA driver compatible with CUDA 11.8

## Setup Instructions

### Step 1: Clone/Copy the Project

Copy the entire project folder to your target machine.

### Step 2: Create Conda Environment

```bash
conda create -n rl_project python=3.11 -y
conda activate rl_project
```

### Step 3: Install Dependencies

```bash
pip install stable-baselines3 gymnasium gymnasium-robotics numpy tensorboard tqdm rich
```

### Step 4: Install PyTorch with CUDA Support

For A5000 GPU (NVIDIA, CUDA compatible):

```bash
pip uninstall torch -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

This installs PyTorch 2.7.1 with CUDA 11.8 support, which is compatible with A5000.

### Step 5: Verify GPU Setup (Optional)

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

## Running the Training

### Activate Environment

```bash
conda activate rl_project
```

### Run Training Script

```bash
python train.py
```

The script will:
1. Create `outputs/models/` and `outputs/logs/` directories
2. Initialize SAC or PPO model (configured in `config.py`)
3. Start training on your GPU
4. Save checkpoints every 50k steps
5. Save the final model

### Monitor Training (Optional)

In another terminal, view TensorBoard logs:

```bash
conda activate rl_project
tensorboard --logdir outputs/logs
```

Then open `http://localhost:6006` in your browser.

## Configuration

Edit `config.py` to change:
- `ALGORITHM`: "SAC" or "PPO"
- `REWARD_TYPE`: "sparse" or "dense"
- `TOTAL_STEPS`: Total training steps (default: 500,000)
- `SAVE_FREQ`: Checkpoint save frequency (default: 50,000)
- Algorithm hyperparameters in `PPO_CONFIG`, `SAC_CONFIG`, `HER_CONFIG`

## Project Structure

```
├── config.py           # Configuration and hyperparameters
├── env_wrapper.py      # Environment wrappers (dense/sparse rewards)
├── train.py           # Main training script
├── test_env.py        # Environment testing script
├── README.md          # This file
└── outputs/
    ├── models/        # Saved model checkpoints
    └── logs/          # TensorBoard logs
```

## Key Files

- **config.py**: Defines algorithm choice, reward type, and all hyperparameters
- **env_wrapper.py**: Wraps the Gymnasium environment with custom reward functions
- **train.py**: Main training loop with callbacks for checkpointing and evaluation

## Troubleshooting

### GPU Not Being Used

If training shows "Using cpu device":
1. Verify CUDA installation: `nvidia-smi`
2. Check PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`
3. Reinstall PyTorch with CUDA:
   ```bash
   pip uninstall torch -y
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### Missing Dependencies

If you get an ImportError, install missing packages:
```bash
conda activate rl_project
pip install [package_name]
```

### Mujoco/Environment Issues

Ensure gymnasium-robotics is installed:
```bash
pip install gymnasium-robotics
```

## Output Files

After training, you'll find:
- `outputs/models/SAC_sparse_[timestamp]/final_model.zip` - Final trained model
- `outputs/models/SAC_sparse_[timestamp]/best_model.zip` - Best checkpoint (highest eval reward)
- `outputs/logs/SAC_sparse_[timestamp]/` - TensorBoard event files

## Notes

- Training on A5000 should be significantly faster than CPU
- First environment interaction may take a moment while MuJoCo initializes
- Sparse reward training with HER is recommended for harder tasks
- Dense reward is simpler but may require more tuning
