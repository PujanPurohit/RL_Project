#!/usr/bin/env python3
import sys
print("Starting test script...", flush=True)

try:
    print("Importing config...", flush=True)
    from config import ENV_ID
    print(f"ENV_ID: {ENV_ID}", flush=True)
    
    print("Importing env_wrapper...", flush=True)
    from env_wrapper import make_env
    print("env_wrapper imported successfully", flush=True)
    
    print("Creating environment...", flush=True)
    env = make_env(reward_type="dense")
    print(f"Environment created: {env}", flush=True)
    
    obs, info = env.reset()
    print(f"Environment reset successful. Obs keys: {obs.keys()}", flush=True)
    
    env.close()
    print("Test successful!", flush=True)
    
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
