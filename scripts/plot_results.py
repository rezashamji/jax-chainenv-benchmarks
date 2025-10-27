#!/usr/bin/env python3
import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-darkgrid")

# consistent look
COLORS = {"ppo":"tab:blue","ddpg":"tab:green","sac":"tab:red","pqn":"tab:orange"}
DIFFICULTIES = ["easy","medium","hard"]
BUDGET = {"easy": 80_000, "medium": 80_000, "hard": 120_000}
MAX_BUDGET = max(BUDGET.values())

# ---------------- helpers ----------------
def load_csv(path):
    df = pd.read_csv(path, header=None)
    if df.shape[1] == 1:
        df.columns = ["return"]
        df["steps"] = np.arange(1, len(df) + 1)
    else:
        df.columns = ["steps", "return"]
    return df

def smooth(y, window=10):
    if len(y) < window:
        return y
    return np.convolve(y, np.ones(window) / window, mode="valid")

# ---------------- plotting functions ----------------
def plot_by_difficulty(base_dir, mode):
    """
    Compare all algorithms at each difficulty level.
    mode: 'train' or 'eval'
    """
    algos = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    fig, axes = plt.subplots(1, len(DIFFICULTIES), figsize=(15, 4), sharey=True)

    for j, diff in enumerate(DIFFICULTIES):
        ax = axes[j]
        budget = BUDGET[diff]
        for algo in algos:
            csv_path = os.path.join(base_dir, algo, f"{diff}{'_eval' if mode=='eval' else ''}.csv")
            if not os.path.exists(csv_path):
                continue
            df = load_csv(csv_path)
            df = df[df["steps"] <= budget]
            if df.empty:
                continue
            y = smooth(df["return"].values, window=8)
            x = df["steps"].values[:len(y)]
            ax.plot(x, y, label=algo.upper(), color=COLORS.get(algo, None), linewidth=1.8)
        ax.set_title(f"{diff.capitalize()} Chain", fontsize=12)
        ax.set_xlabel("Environment Steps")
        if j == 0:
            ax.set_ylabel("Episodic Return")
        ax.set_xlim(0, budget)
        ax.legend(fontsize=8)

    title_suffix = "Deterministic Evaluation" if mode == "eval" else "Training (ε-greedy)"
    fig.suptitle(f"Exploration Benchmark: ChainEnv ({title_suffix})", fontsize=14)
    plt.tight_layout()
    out = f"runs/chainenv_{mode}_by_difficulty.png"
    plt.savefig(out, dpi=200)
    print(f" Saved {out}")

def plot_by_algorithm(base_dir, mode):
    """
    For each algorithm, show performance across easy/medium/hard.
    mode: 'train' or 'eval'
    """
    algos = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    fig, axes = plt.subplots(len(algos), 1, figsize=(8, 2.8 * len(algos)), sharex=True)

    for i, algo in enumerate(algos):
        ax = axes[i] if len(algos) > 1 else axes
        for diff in DIFFICULTIES:
            csv_path = os.path.join(base_dir, algo, f"{diff}{'_eval' if mode=='eval' else ''}.csv")
            if not os.path.exists(csv_path):
                continue
            df = load_csv(csv_path)
            df = df[df["steps"] <= BUDGET[diff]]
            y = smooth(df["return"].values, window=8)
            x = df["steps"].values[:len(y)]
            ax.plot(x, y, label=diff.capitalize(), linewidth=2)
        ax.set_title(f"{algo.upper()} Across Difficulties", fontsize=12, color=COLORS.get(algo, "black"))
        ax.set_ylabel("Episodic Return")
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Environment Steps")
    axes[-1].set_xlim(0, MAX_BUDGET)
    title_suffix = "Deterministic Evaluation" if mode == "eval" else "Training (ε-greedy)"
    fig.suptitle(f"ChainEnv Difficulty Scaling per Algorithm ({title_suffix})", fontsize=14)
    plt.tight_layout()
    out = f"runs/chainenv_{mode}_by_algorithm.png"
    plt.savefig(out, dpi=200)
    print(f" Saved {out}")

# ---------------- main ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ChainEnv benchmark results")
    parser.add_argument("--mode", choices=["train", "eval"], default="eval",
                        help="Choose whether to plot training or evaluation curves")
    args = parser.parse_args()

    base_dir = "runs"
    plot_by_difficulty(base_dir, args.mode)
    plot_by_algorithm(base_dir, args.mode)
    plt.show()
