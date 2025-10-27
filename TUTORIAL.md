# TUTORIAL.md

## 0) Goal

By the end, you’ll understand:
1) How ChainEnv works and why it’s a good exploration testbed.  
2) The full training loop and logging pipeline.  
3) What each algorithm does (PPO, DDPG, SAC, PQN) and how the code maps to the math.  
4) How to run, plot, and extend experiments.

---

## 1) The environment (deep dive)

**State**: a single integer position `s ∈ {0,…,N-1}` wrapped as `obs = [float(s)]`.  

**Action → move**:
- **PPO/PQN**: `action ∈ {0,1}`, map to `-1` (left) or `+1` (right).
- **DDPG/SAC**: `action ∈ ℝ`, env uses `action > 0 → +1`, else `-1`.

**Slip**: with probability `slip`, flip the chosen move (harder exploration).

**Reward**:
- `r_small` at position 1 (local optimum/lure).
- `r_big` at the goal `N-1` (sparse terminal reward).

**Done**: reaching goal or hitting horizon `H`.

Pure JAX, so `reset/step` are side-effect free and work with `jit`, `vmap`, `lax.scan`.

Vectorization:
- `batch_reset(keys, params)` → run many envs in parallel.
- `batch_step(states, actions, params)` → parallel step.

Presets (`envs/chain_jax_env.py → DIFFICULTIES`):
````

easy   : N=5,  H=15, slip=0.00, r_small=0.3, r_big=1.0  (max greedy ≈ 1.3)
medium : N=7,  H=20, slip=0.15, r_small=0.1, r_big=1.0  (max greedy ≈ 1.1)
hard   : N=9,  H=25, slip=0.25, r_small=0.0, r_big=1.0  (max greedy ≈ 1.0)

````
> **Note:** On Easy, total return can exceed `r_small + r_big = 1.3` if the policy revisits position `1` multiple times before reaching the goal.
---

## 2) JAX patterns used

- **PRNG**: pass and split keys (`key, subk = jax.random.split(key)`).
- **Pure functions**: env/learners return new states (no mutation).
- **Vectorization**: `vmap` for many envs at once.
- **JIT**: compile heavy update steps once.
- **Device-friendly buffers**: see `algorithms/jax_buffer.py` for `.at[idx].set(...)` patterns.

---

## 3) Training pipeline (all algorithms)

1. **Initialize** env batch and learner networks (dummy shapes → init).
2. **Collect** actions:
   - **On-policy** (PPO, PQN): act with the **current** behavior (categorical sampling for PPO; ε-greedy for PQN).
   - **Off-policy** (DDPG, SAC): act with exploration noise (Gaussian for DDPG; tanh-Gaussian sampling for SAC) and learn from a **replay buffer** (simple Python list here; optional JAX ring buffer provided).
3. **Step envs** in parallel (`vmap(step, ...)`), record rewards/dones.
4. **Log episodic returns** when any env finishes; reset only those envs.
5. **Update** the learner:
   - **PPO**: multiple epochs/minibatches over the on-policy rollout, with **clipped ratios** + **GAE** + entropy.
   - **DDPG/SAC**: sample a minibatch from replay and do one off-policy update (twin Q + targets; SAC also includes the **α logπ** term).
   - **PQN**: compute **Q(λ)** targets via reverse scan, then SGD on `Q(s,a)` vs λ-target (RAdam + grad-clip).
6. **Repeat** until `CHAIN_TOTAL_ENV_STEPS`.
7. **Evaluation** (greedy):
   - **DDPG/SAC**: deterministic actions (DDPG actor mean; SAC uses `tanh(μ)`).
   - **PPO**: greedy via `pi.mode()` (note: current script writes the same greedy curve to both train and eval CSVs).
   - **PQN**: both ε-greedy **training** and separate **greedy eval** curves are saved.

---

## 4) Algorithms: what & where in the code

### 4.1 PPO — on-policy policy gradient
- **File**: `algorithms/ppo_chain_jax.py`
- **Policy**: categorical over 2 actions via `distrax.Categorical(logits=...)`.
- **Rollout**: `lax.scan` for `NUM_STEPS`, storing `(obs, action, log_prob, value, reward, done)`.
- **GAE**:
  \[
  \delta_t = r_t + \gamma V(s_{t+1})(1-d_t) - V(s_t),\quad
  \hat A_t = \delta_t + \gamma\lambda(1-d_t)\hat A_{t+1}
  \]
- **Objective**: clipped ratio policy loss + clipped value loss − entropy bonus.
- **Eval**: `pi.mode()` for greedy left/right.

### 4.2 DDPG — off-policy deterministic actor
- **Files**: `algorithms/ddpg_chain_jax.py`, `external/jaxrl_ddpg/*`
- **Exploration**: **Gaussian** noise added to μ(s) (the paper used OU; Gaussian is common in practice).
- **Critics**: **twin Q** heads; use `min(Q1,Q2)` (clipped double-Q).
- **Targets**: **Polyak** for critic and **target actor**.
- **Critic target**:
  \[
  y = r + \gamma\,\text{mask}\cdot Q_{\text{tgt}}(s', \mu_{\text{tgt}}(s'))
  \]
- **Actor update**:
  \[
  \max_\theta~\mathbb{E}_s[Q_\phi(s,\mu_\theta(s))]\; \Leftrightarrow\; \min_\theta -Q_\phi(s,\mu_\theta(s))
  \]

### 4.3 SAC — off-policy stochastic actor (maximum entropy)
- **Files**: `algorithms/sac_chain_jax.py`, `external/jaxrl2_sac/*`
- **Policy**: reparameterized **tanh-Gaussian**.
- **Target with entropy**:
  \[
  y = r + \gamma\,\text{mask}\cdot \big(\min(Q_{\text{tgt}}(s',a')) - \alpha \log\pi(a'|s')\big)
  \]
- **Actor loss**:
  \[
  \min_\theta~\mathbb{E}_s\big[\alpha \log\pi_\theta(a|s) - \min(Q_\phi(s,a))\big]
  \]
- **Note**: implementation here uses **fixed α** (configurable).

### 4.4 PQN — on-policy Q-learning with Q(λ)
- **Files**: `algorithms/pqn_chain_jax.py`, `external/purejaxql_pqn/pqn_learner.py`
- **Policy**: ε-greedy over `Q(s,a)`.
- **Q(λ) (reverse scan)**:
  Intuitively, push the final reward signal back over time using λ-returns:
  \[
  \text{boot}_t = r_t + \gamma(1-d_t)\max_a Q(s_t,a),\quad
  \lambda\text{-ret}_t = \text{boot}_t + \gamma\lambda(\lambda\text{-ret}_{t+1} - \max_a Q(s_t,a))
  \]
  (with terminals cutting the recursion).
- **Learner**: MSE on `Q(s_t,a_t)` vs λ-target; **RAdam** + grad-clip.

---

## 5) Running experiments

**Single run**
```bash
export CHAIN_DIFFICULTY=hard
export CHAIN_TOTAL_ENV_STEPS=120000
export CHAIN_SEED=0
python algorithms/sac_chain_jax.py
````

**Batch runs**

```bash
python scripts/run_experiments.py --clear
# or a subset:
python scripts/run_experiments.py \
  --algos ppo pqn \
  --difficulties medium hard \
  --budget 80000 120000 \
  --seed 0 --clear
```

**Override env parameters**

```bash
python scripts/run_experiments.py --algos sac \
  --difficulties medium \
  --override N=11 H=30 SLIP=0.2 R_SMALL=0.05 R_BIG=1.0
```

> Overrides are exported as `CHAIN_<KEY>`; add an env-read shim if you want the env to ingest them directly.

---

## 6) Plotting & logs

* CSV: `steps,return` (plotter also tolerates single-column “return”).
* `scripts/plot_results.py`

  * `--mode eval` prefers `_eval.csv`.
  * `--mode train` uses training CSVs.
  * Smoothing: moving average (window=8).

Produces:

* `runs/chainenv_eval_by_difficulty.png`
* `runs/chainenv_eval_by_algorithm.png`
* (and `train_...` variants)

> **PPO** currently writes the same greedy curve to **both** train and eval CSVs.

---

## 7) Extending

1. Create `algorithms/myalgo_chain_jax.py`.
2. Follow the pattern:

   * Read env via `get_run_config()`.
   * Vectorize env (`vmap(reset)`, `vmap(step)`).
   * Implement learner; log `(steps,return)` pairs.
   * Optional: write `_eval.csv` for greedy evaluation.
3. Add to `scripts/run_experiments.py → ALGORITHMS`.
4. The plotter picks it up automatically.

---

## 8) Troubleshooting

* **JAX install**: ensure the correct wheel for your accelerator.
* **Shapes**: observations are `(B,1)`; DDPG/SAC pass `actions[:, 0]` into `step`.
* **Masks vs done**: learners expect `mask = 1.0 - done`.
* **First run is slow**: JIT compilation.
* **Flat curves**:

  * Increase exploration: **EXPL_NOISE** (DDPG), **ALPHA**/target entropy (SAC), **ε / slower decay** (PQN).
  * Increase horizon/budget: `H`, `CHAIN_TOTAL_ENV_STEPS`.

```