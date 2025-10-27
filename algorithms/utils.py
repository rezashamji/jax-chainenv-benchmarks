# algorithms/utils.py
import os

def get_run_config():
    difficulty = os.getenv("CHAIN_DIFFICULTY", "medium")
    # Back-compat: fall back to CHAIN_TOTAL_STEPS if present
    total_env_steps = int(os.getenv("CHAIN_TOTAL_ENV_STEPS",
                           os.getenv("CHAIN_TOTAL_STEPS", "80000")))
    seed = int(os.getenv("CHAIN_SEED", "0"))
    return difficulty, total_env_steps, seed