# TUTORIAL.md

## 0) Goal of this tutorial

By the end, you’ll understand:

1. How ChainEnv works and why it’s a good exploration testbed.
2. The full training loop and logging pipeline.
3. What each algorithm does (PPO, DDPG, SAC, PQN) and **how the code maps to the math**.
4. How to run, plot, and extend experiments.

---

## 1) The environment (deep dive)

**State**: a single integer position `s ∈ {0,…,N-1}` wrapped as `obs = [float(s)]`.

**Action → move**:
- For PPO/PQN: `action ∈ {0,1}`, map to `-1` (left) or `+1` (right).
- For DDPG/SAC: `action ∈ ℝ`, map to `-1` if `a ≤ 0`, else `+1`.

**Slip**: with probability `slip`, we flip the chosen move (makes exploration harder).

**Reward**:
- `r_small` at position 1 (a tempting local optimum).
- `r_big` at the goal `N-1` (sparse but final).

**Done**: reaching goal or hitting horizon `H`.

The code is 100% JAX: no side effects; `reset/step` are pure functions, enabling `jit`, `vmap`, and `lax.scan`.

Vectorization wrappers:
- `batch_reset(keys, params)` → run many envs in parallel.
- `batch_step(states, actions, params)` → parallel step.

Presets (`envs/chain_jax_env.py → DIFFICULTIES`):
```
easy   : N=5,  H=15, slip=0.00, r_small=0.3, r_big=1.0
medium : N=7,  H=20, slip=0.15, r_small=0.1, r_big=1.0
hard   : N=9,  H=25, slip=0.25, r_small=0.0, r_big=1.0

```

---

## 2) JAX patterns used

- **PRNG**: pass and split keys (`key, subk = jax.random.split(key)`).
- **Pure functions**: env/learners return new states rather than mutating in place.
- **Vectorization**: `jax.vmap` to step many envs at once.
- **JIT**: heaviest update steps are compiled once, then run fast.
- **.at[idx].set(...)**: JAX-friendly “in-place” writes for buffers (see `algorithms/jax_buffer.py`).

---

## 3) Training pipeline (all algorithms)

1. **Initialize** env batch and learner networks (shapes are derived from a dummy obs/action).
2. **Collect** actions:
   - On-policy (PPO, PQN): act with the **current** policy.
   - Off-policy (DDPG, SAC): act with the current policy and learn from a **replay buffer** (simple Python list for clarity; `jax_buffer.py` provides a device-resident ring buffer).
3. **Step envs** in parallel (`vmap(step, ...)`), record rewards/dones.
4. **Log episodic returns** whenever any env finishes; reset only those envs.
5. **Update** the learner:
   - PPO: epoch/minibatch updates over the on-policy rollout.
   - DDPG/SAC: sample a random minibatch from the buffer and do one off-policy update.
   - PQN: compute **Q(λ)** targets via a reverse scan, then SGD on Q.
6. **Repeat** until `CHAIN_TOTAL_ENV_STEPS` is reached.
7. **Evaluation**:
   - DDPG/SAC: **deterministic eval** (no noise; “greedy action”) to get a clean curve.
   - PPO: use `pi.mode()` over the learned categorical for a clean greedy curve. (Current script writes this deterministic curve to both train and eval CSVs.)
   - PQN: training curve reflects greedy behavior as ε decays; we also save a separate greedy eval curve.

---

## 4) Algorithms: what & where in the code

### 4.1 PPO (on-policy policy gradient)
- **Files**: `algorithms/ppo_chain_jax.py`
- **Policy**: categorical over 2 actions (left/right).
  - `ActorCritic.__call__` builds `pi = distrax.Categorical(logits=...)` and a value head.
- **Rollout**: `NUM_STEPS` via `_env_step` inside a `lax.scan`, storing `(obs, action, log_prob, value, reward, done)`.
- **Advantage (GAE)**:
  ```python
  delta = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
  GAE_t = delta + γ * λ * (1 - done_t) * GAE_{t+1}
  ```

Implemented in `_calculate_gae` with a reverse scan.

* **Objective**: **clipped ratio** policy loss + clipped value loss − entropy bonus.
* **Training**: multiple `UPDATE_EPOCHS × NUM_MINIBATCHES`.
* **Eval**: `pi.mode()` gives greedy left/right.

### 4.2 DDPG (off-policy deterministic actor)

* **Files**: `algorithms/ddpg_chain_jax.py`, `external/jaxrl_ddpg/*`
* **Exploration**: **Gaussian** noise added to the deterministic actor in `sample_actions`.
* **Critics**: **twin Q** heads; use `min(Q1, Q2)` to reduce overestimation.
* **Targets**: **Polyak averaging** of critic (and actor target in the fused learner).
* **Critic target**:

  ```
  y = r + γ * mask * Q_target1( s', μ_target(s') )
  ```

  (`mask = 1 - done`)
* **Actor update**:

  ```
  maximize Q1(s, μ(s))  →  minimize -Q1(s, μ(s))
  ```

### 4.3 SAC (off-policy stochastic actor with entropy)

* **Files**: `algorithms/sac_chain_jax.py`, `external/jaxrl2_sac/*`
* **Policy**: **tanh-Gaussian** (reparameterized).
* **Target with entropy**:

  ```
  y = r + γ * mask * [ min_target_Q(s', a') - α * logπ(a'|s') ],  a' ~ π(.|s')
  ```
* **Actor loss**:

  ```
  minimize  E_s [ α * logπ(a|s) - min_Q(s, a) ]
  ```
* **Eval**: deterministic (`tanh(μ)`) for clean curves.

### 4.4 PQN (on-policy Q-learning with Q(λ))

* **Files**: `algorithms/pqn_chain_jax.py`, `external/purejaxql_pqn/pqn_learner.py`
* **Policy**: ε-greedy over `Q(s,a)`.
* **Q(λ) via reverse scan** (pushes final rewards back fast):

  ```python
  boot  = r_t + γ * (1 - done_t) * max_a Q(s_t, a)
  delta = lamret_next - max_a Q(s_t, a)
  lamret = boot + γ * λ * delta
  lamret = (1 - done_t) * lamret + done_t * r_t
  ```
* **Learner**: MSE on `Qθ(s_t, a_t)` vs λ-target; RAdam + grad-clip.

---

## 5) Running experiments

### Single run (explicit)

```bash
# Example: SAC on hard for 120k steps
export CHAIN_DIFFICULTY=hard
export CHAIN_TOTAL_ENV_STEPS=120000
export CHAIN_SEED=0
python algorithms/sac_chain_jax.py
```

### Batch runs (recommended)

```bash
# All algos × all difficulties with default budgets
python scripts/run_experiments.py --clear

# Only PPO and PQN, medium/hard, custom budgets, fixed seed
python scripts/run_experiments.py \
  --algos ppo pqn \
  --difficulties medium hard \
  --budget 80000 120000 \
  --seed 0 --clear
```

### Override env parameters

```bash
# Make the chain longer with more slip (harder exploration)
python scripts/run_experiments.py --algos sac \
  --difficulties medium \
  --override N=11 H=30 SLIP=0.2 R_SMALL=0.05 R_BIG=1.0
```

> The runner passes these as `CHAIN_<KEY>` env vars. The current env uses the built-in presets; to *apply* overrides, extend the entrypoints or `chain_jax_env.py` to read those vars into `EnvParams`.

---

## 6) Plotting and reading logs

* CSV format: `steps,return` (the plotter also tolerates a single “return” column).
* `scripts/plot_results.py`:

  * `--mode eval` prefers `_eval.csv` where available.
  * `--mode train` uses training CSVs.
  * Smoothing: moving average (window=8).

Produced figures:

* `runs/chainenv_eval_by_difficulty.png` : Algorithms compared within each difficulty.
* `runs/chainenv_eval_by_algorithm.png`: Difficulties compared within each algorithm.
* `runs/chainenv_train_by_difficulty.png`
* `runs/chainenv_train_by_algorithm.png`

> PPO’s current script writes the same deterministic curve to both train and eval CSVs.

---

## 7) Extending the code

### Add a new algorithm

1. Create `algorithms/myalgo_chain_jax.py`.
2. Follow the pattern:

   * Read env via `get_run_config()`.
   * Vectorize env (`vmap(reset)`, `vmap(step)`).
   * Implement your learner; log `(steps,return)` pairs.
   * Optional: write `_eval.csv` for greedy evaluation.
3. Add it to `scripts/run_experiments.py → ALGORITHMS`.
4. The plotter will automatically pick it up.

### Use the JAX ring buffer

See `algorithms/jax_buffer.py` for a device-resident replay that returns JAX arrays:

* `JaxReplayBuffer(capacity, obs_dim, act_dim)`
* `add_batch(obs, act, rew, mask, next_obs)`
* `sample(key, batch_size) → (idx, batch_dict)`

---

## 8) Troubleshooting

* **JAX install**: If wheels fail, you likely need the correct jax/jaxlib build for your accelerator/OS.
* **Shapes**: The env expects `(B,1)` observations and a scalar action per env. For DDPG/SAC, we pass `actions[:, 0]` into `step`.
* **Masks vs done**: learners expect `mask = 1.0 - done` to cut bootstrapping at terminals.
* **Slow first run**: JIT compilation happens once and then speeds up.
* **Flat lines in curves**:

  * Too little exploration? Increase `EXPL_NOISE` (DDPG), `ALPHA` (SAC), or ε / decay ratio (PQN).
  * Too short horizon/budget? Increase `H` or `CHAIN_TOTAL_ENV_STEPS`.

---

## 9) FAQ

**SAC is “stochastic”: why is eval deterministic?**
Training samples from the tanh-Gaussian; evaluation uses `tanh(μ)` (no noise) to show best-guess policy performance.

**What’s the “mask”?**
`mask = 1.0 - done`. If `done=1`, we **cut** bootstrapping; otherwise we allow the next-state value.

**Why 80k steps (or 120k for hard)?**
Convenient budgets that let algorithms explore enough to reach the final goal given the slip/horizon.

**When should I prefer PQN here?**
On **sparse reward** chain tasks, PQN’s **Q(λ)** targets propagate final rewards backward fast: great for “learn to always go right.”