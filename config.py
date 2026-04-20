ALGORITHM   = "SAC"     
REWARD_TYPE = "sparse"    

ENV_ID      = "HandManipulateBlockRotateXYZ-v1"
TOTAL_STEPS = 500_000
SAVE_FREQ   = 50_000

PPO_CONFIG = dict(
    learning_rate = 3e-4,
    n_steps       = 2048,
    batch_size    = 64,
    n_epochs      = 10,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    ent_coef      = 0.01,
    verbose       = 1,
)

SAC_CONFIG = dict(
    learning_rate    = 3e-4,
    buffer_size      = 500_000,
    learning_starts  = 1000,
    batch_size       = 256,
    tau              = 0.005,
    gamma            = 0.99,
    train_freq       = 1,
    gradient_steps   = 1,
    ent_coef         = "auto",
    verbose          = 1,
)

HER_CONFIG = dict(
    n_sampled_goal          = 4,
    goal_selection_strategy = "future",
)