#scripts/run_experiments.py
#!/usr/bin/env python3
# ==========================================================
# Dynamic ChainEnv experiment runner
#   - Choose specific algorithms, difficulties, and budgets
#   - Pass overrides for environment parameters or seeds
# ==========================================================
import os
import subprocess
import shutil
import argparse
from datetime import datetime

ALGORITHMS = {
    "ppo": "algorithms/ppo_chain_jax.py",
    "ddpg": "algorithms/ddpg_chain_jax.py",
    "sac": "algorithms/sac_chain_jax.py",
    "pqn": "algorithms/pqn_chain_jax.py",
}

DIFFICULTIES = ["easy", "medium", "hard"]
DEFAULT_BUDGET = {"easy": 80_000, "medium": 80_000, "hard": 120_000}


def run_experiment(algo, difficulty, steps, seed, overrides):
    print(f"\n[▶] Running {algo.upper()} | diff={difficulty} | steps={steps} | seed={seed}")

    env = os.environ.copy()
    env["CHAIN_DIFFICULTY"] = difficulty
    env["CHAIN_TOTAL_ENV_STEPS"] = str(steps)
    env["CHAIN_SEED"] = str(seed)

    # Optional overrides: chain length, slip, etc.
    if overrides:
        for k, v in overrides.items():
            env[f"CHAIN_{k.upper()}"] = str(v)

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = root_dir

    subprocess.run(["python3", ALGORITHMS[algo]], env=env, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run ChainEnv RL benchmarks dynamically"
    )
    parser.add_argument(
        "--algos", nargs="+", default=list(ALGORITHMS.keys()),
        help=f"Which algorithms to run (default: all) — choices: {list(ALGORITHMS.keys())}",
    )
    parser.add_argument(
        "--difficulties", nargs="+", default=DIFFICULTIES,
        help=f"Which difficulties to run (default: all) — choices: {DIFFICULTIES}",
    )
    parser.add_argument(
        "--budget", nargs="+", type=int,
        help="Custom env-step budgets (override default per difficulty). Provide one per difficulty in order.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clear", action="store_true", help="Clear existing runs/ folder first")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Env param overrides, e.g. --override N=9 H=25 SLIP=0.3 R_SMALL=0.1")
    args = parser.parse_args()

    # Build overrides dict if any
    overrides = {}
    for pair in args.override:
        if "=" in pair:
            k, v = pair.split("=", 1)
            overrides[k] = v

    # Manage output folder
    if args.clear and os.path.exists("runs"):
        shutil.rmtree("runs")
        print(" Cleared existing 'runs/' directory...")
    os.makedirs("runs", exist_ok=True)

    budgets = dict(DEFAULT_BUDGET)
    if args.budget:
        for i, diff in enumerate(args.difficulties):
            budgets[diff] = args.budget[min(i, len(args.budget)-1)]

    start_time = datetime.now()

    for algo in args.algos:
        for diff in args.difficulties:
            steps = budgets[diff]
            run_experiment(algo, diff, steps, args.seed, overrides)

    total_time = (datetime.now() - start_time).total_seconds()
    print(f"\n🎉 Experiments finished in {total_time/60:.1f} minutes.")
    print("Results saved under runs/<algo>/<difficulty>.csv")


if __name__ == "__main__":
    main()
