# README.md 

## ChainEnv RL Benchmarks (JAX)

A complete RL playground around a 1-D **ChainEnv** with tunable exploration difficulty. Includes JAX implementations of **PPO**, **DDPG**, **SAC**, and **PQN (Q(λ))**, vectorized environments, a dynamic experiment runner, optional JAX ring buffers, and plotting utilities.

## Why this repo?
- **Compare exploration** on a sparse-reward toy task.
- **Learn JAX RL patterns** (jit, vmap, PRNG, pure functions).
- **Start simple, extend easily** with clear learners.

---

## Quickstart

### 1) Environment
```bash
python -m venv .venv && source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install --upgrade pip

# Core deps
pip install jax jaxlib flax optax distrax numpy matplotlib pandas
````

> Tip: GPU/TPU wheels vary by platform. Install the correct `jax/jaxlib` wheels for your accelerator.

### 2) Run one algorithm

```bash
# Difficulty, total env steps, and seed via env vars
export CHAIN_DIFFICULTY=medium
export CHAIN_TOTAL_ENV_STEPS=80000
export CHAIN_SEED=0

# PPO example
python algorithms/ppo_chain_jax.py
```

Outputs go to `runs/ppo/medium.csv` and `runs/ppo/medium_eval.csv` (when applicable).

### 3) Run a full sweep

```bash
python scripts/run_experiments.py \
  --algos ppo ddpg sac pqn \
  --difficulties easy medium hard \
  --seed 0 \
  --clear
```

> If you omit `--algos`, `--difficulties`, or `--seed`, sensible defaults run everything. (So `python scripts/run_experiments.py --clear` is equivalent to the example above.)

### 4) Plot results

```bash
# default: eval curves; pass --mode train for training curves
python scripts/plot_results.py
# Produces:
# runs/chainenv_eval_by_difficulty.png
# runs/chainenv_eval_by_algorithm.png
# (and train_... variants if --mode train)
```

---

## Repository layout

```
algorithms/
  ppo_chain_jax.py          # On-policy PPO (categorical policy + GAE)
  ddpg_chain_jax.py         # Off-policy DDPG (deterministic actor + twin Q)
  sac_chain_jax.py          # Off-policy SAC (tanh-Gaussian + twin Q)
  pqn_chain_jax.py          # On-policy PQN with Q(λ) targets
  jax_buffer.py             # Optional ring buffers (JAX arrays)
  __init__.py, utils.py     # Small helpers (env var config)

envs/
  chain_jax_env.py          # Functional, vectorized ChainEnv (JAX)
  __init__.py

external/
  jaxrl_ddpg/               # Minimal DDPG learner (actor/critic/Polyak)
  jaxrl2_sac/               # Minimal SAC learner (tanh-Gaussian, α fixed)
  purejaxql_pqn/            # PQN learner (Flax + Optax RAdam)

scripts/
  run_experiments.py        # CLI runner for grid sweeps
  plot_results.py           # Plots by difficulty/algorithm
```

---

## The environment (ChainEnv)

* **Obs**: `(1,)` current position (float).
* **Actions**:

  * PPO/PQN: discrete `{0:left, 1:right}` (PPO samples; PQN ε-greedy).
  * DDPG/SAC: continuous scalar; `action > 0 → right`, else left.
* **Stochasticity**: with probability `slip`, flip the chosen direction.
* **Rewards**: `r_small` at position 1 (local lure), `r_big` at goal (`N-1`).
* **Done**: reaching goal or `t >= H`.

Presets (`envs/chain_jax_env.py → DIFFICULTIES`):

```
easy:   N=5,  H=15, slip=0.00, r_small=0.3, r_big=1.0
medium: N=7,  H=20, slip=0.15, r_small=0.1, r_big=1.0
hard:   N=9,  H=25, slip=0.25, r_small=0.0, r_big=1.0
```

---

## Outputs & logging

Each algorithm writes CSVs under `runs/<algo>/<difficulty>.csv`:

* Two columns: `steps,return` (episodic return vs env steps).
* `_eval.csv` files are deterministic “greedy” evaluations when available:

  * **DDPG/SAC**: always write `<difficulty>_eval.csv`.
  * **PPO**: current script writes the same deterministic curve to both train and eval CSVs.
  * **PQN**: saves training (ε-greedy) and a separate greedy eval curve.

`plot_results.py`:

* **By difficulty**: overlays algorithms per difficulty (fixed budgets).
* **By algorithm**: shows easy/medium/hard curves stacked per algorithm.
* Smoothing: moving average (window=8).

---

## Useful env vars (read in `algorithms/utils.py`)

* `CHAIN_DIFFICULTY ∈ {easy,medium,hard}` (default: `medium`)
* `CHAIN_TOTAL_ENV_STEPS` (default: `80000`)
* `CHAIN_SEED` (default: `0`)

The runner also accepts overrides like:

```
--override N=9 H=25 SLIP=0.3 R_SMALL=0.1
```

> These are exported as `CHAIN_<KEY>` env vars; use presets by default. (If you want overrides to alter the env beyond presets, add a small read-from-env shim in `chain_jax_env.py`.)

---

## Key algorithms (one-liners)

* **PPO** (on-policy): stochastic categorical policy, **ratio clipping**, **GAE**, entropy bonus.
* **DDPG** (off-policy): deterministic actor + **twin Q** critics, **target networks**, Gaussian exploration.
* **SAC** (off-policy): tanh-Gaussian policy, entropy-regularized Q targets, **twin Q + targets**.
* **PQN** (on-policy): **Q-network** trained with **Q(λ)** targets that push final rewards back through time.

---

## Extending

* Add a new file in `algorithms/`.
* Log a CSV like the others (`steps,return`).
* Register it in `scripts/run_experiments.py → ALGORITHMS`.
* Reuse `envs/chain_jax_env.py` and `algorithms/jax_buffer.py` as needed.
