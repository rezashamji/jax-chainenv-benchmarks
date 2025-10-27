# CONCLUSIONS.md

## Figures referenced

The following plots visualize the summarized results discussed below.  
All figures are located in the `results/` folder of this repository.

| Figure | Description |
|:--|:--|
| **chainenv_train_by_algorithm.png** | Training curves (ε-greedy) grouped **by algorithm** across all difficulty levels. |
| **chainenv_train_by_difficulty.png** | Training curves (ε-greedy) grouped **by difficulty** across all algorithms. |
| **chainenv_eval_by_algorithm.png** | Deterministic evaluation (greedy) grouped **by algorithm**. |
| **chainenv_eval_by_difficulty.png** | Deterministic evaluation (greedy) grouped **by difficulty**. |

These correspond respectively to the **training** (exploratory) and **evaluation** (greedy) phases described later in the conclusions.

---

## What the results show (at a glance)

**Easy chain:**

* **SAC** and **PQN** reliably reach the **final (big) reward**; deterministic eval plateaus near the task’s empirical max (~2.4).
* **PPO** and **DDPG** hover around the **local lure** (≈1.2–1.3) and rarely commit to the long rightward trajectory.

**Medium chain:**

* No algorithm consistently solves the task.
* **PPO** is the least-bad (≈1.1–1.2), while **DDPG ≈ SAC ≈ PQN** settle near suboptimal returns (~0.9–1.0).

**Hard chain:**

* **None** of the algorithms solve it within the current budgets.
* **PQN** shows **rare spikes** (occasional discoveries) but cannot convert them into stable policy improvement; **SAC** and **DDPG** are near-zero; **PPO** drifts around the small-reward baseline with volatility.

These patterns are consistent across **training curves (ε-greedy / noisy)** and **deterministic evaluation curves (greedy)**. The training plots are noisier, especially for PQN, because exploration noise (ε-greedy or stochastic policies) is active; evaluation removes that noise and reveals the actual learned policy.

---

## Per-algorithm takeaways

### PPO (on-policy, clipped PG with GAE)

* **Easy:** gets stuck at the **local optimum**, expected for an on-policy method with limited exploration pressure.
* **Medium/Hard:** also stuck; slight advantage on **Medium** likely comes from decent value learning for the small-reward path plus stable on-policy updates.
* **Why:** On-policy data collection rarely visits the sparse terminal reward; clipping+GAE stabilizes learning but doesn’t create new exploratory behavior.

### DDPG (off-policy, deterministic actor + twin Q)

* **Easy:** tracks the small reward, rarely discovers the long rightward path.
* **Medium/Hard:** fails to explore; value targets stay near the local reward manifold.
* **Why:** Deterministic policies with simple Gaussian noise are notoriously **exploration-poor** on sparse tasks; without hits to the goal, the replay buffer never contains informative targets.

### SAC (off-policy, stochastic actor with entropy)

* **Easy:** **solves**: entropy regularization sustains broad exploration long enough to find and exploit the goal; eval is clean and high.
* **Medium/Hard:** entropy is **not enough**; still fails to discover the long sparse sequence reliably within the budget.
* **Why:** SAC’s exploration is strong but not magic; in harder settings (longer horizon + slip) the probability of stumbling into the full rightward trajectory drops too low for the current budgets/hparams.

### PQN (on-policy Q-learning with **Q(λ)**)

* **Easy:** **solves cleanly** in evaluation; training shows dips because ε-greedy keeps injecting exploration even after the policy is good.
* **Medium:** plateaus suboptimally; **λ-returns can only help after a few successes**, which seem too rare here.
* **Hard:** exhibits **occasional spikes** (some discoveries) but cannot convert them into stable improvement before ε decays and/or before the budget ends.
* **Why:** The strength of PQN is **fast reward propagation once a success occurs** (λ floods credit back). The bottleneck here is **discovery**, not propagation.

---

## Why this is happening (mechanistic read)

1. **Exploration barrier dominates.**
   ChainEnv requires a long sequence of right moves while resisting the attractive small reward at position 1. As horizon increases and **slip** grows, the probability of ever observing the terminal reward **decays sharply**. Algorithms differ mostly in how often they **ever** hit the goal, not in how well they optimize once they see it.

2. **On-policy vs off-policy dynamics.**

   * **On-policy** (PPO, PQN) explores with the **current** behavior; if that behavior locks onto the small reward, improvements stall.
   * **Off-policy** (DDPG, SAC) can in principle learn from **lucky, rare** successes if they ever land in replay but without those, targets never rise above the small-reward plateau.

3. **Why SAC beats DDPG on Easy.**
   SAC’s entropy term sustains **wider action coverage**, so it’s more likely to experience the full rightward trajectory early. Once a few successes appear, twin-Q + targets lock in the better policy. DDPG’s exploration is too myopic here.

4. **Why PQN shines on Easy but not on Medium/Hard.**
   **Q(λ)** gives **excellent credit assignment** *after* a success, but it does not make that first success more likely by itself. With a few hits (Easy), PQN rockets to optimal. With almost none (Medium/Hard), it looks average or worse.

5. **Training vs Eval discrepancy (especially PQN).**
   Training curves include **ε-greedy** actions → visible drops even after learning. Eval is **greedy** → shows the true competency of the learned Q-policy.

---

## Alignment with the papers

* **PPO (Schulman et al., 2017):**
  Paper emphasizes stable policy optimization with clipped ratios, not solving sparse exploration by itself. **Observed behavior aligns**: stable but prone to local optima without explicit exploration bonuses (curiosity, counts, etc.).

* **DDPG (Lillicrap et al., 2015/2016):**
  Strong results on **dense-reward** continuous control; widely known to be brittle in sparse regimes. **Observed behavior aligns**: poor exploration prevents progress.

* **SAC (Haarnoja et al., 2018):**
  Robust across many benchmarks; entropy helps exploration but does not guarantee success on **long-horizon, sparse** tasks. **Observed behavior aligns**: solves Easy, struggles on Medium/Hard without extra help.

* **PQN (Q(λ) on-policy Q-learning):**
  Theory/intuition: once terminal rewards are observed, **λ-returns propagate signal quickly** and reduce bootstrap myopia. **Observed behavior partially aligns**: great on Easy (many terminal hits), limited on Medium/Hard where discovery is the bottleneck.

**Bottom line:** there is **no contradiction** with the literature. The results stress that **exploration, not optimization**, limits performance on ChainEnv as difficulty grows.

---

## Caveats & likely contributors to gaps

* **Action-space mismatch for DDPG/SAC.** We learn continuous actions but **threshold to {left,right}**. That discards gradient structure (sign only) and may blunt learning. Consider **SAC-Discrete** or native discrete critics/policies.

* **Hyperparameters are generic.**
  SAC temperature (α), DDPG noise scale/schedule, PPO entropy bonus, PQN ε-schedule and λ may be **under-tuned for sparse exploration**.

* **Single-seed sensitivity.**
  ChainEnv is extremely sensitive to the **first few thousand steps**. Multi-seed (e.g., 10 seeds) with mean ± 95% CI would give a more faithful picture.

* **Return scale and logging.**
  Reward magnitudes (r_small vs r_big) and horizon interact with critic targets; normalizing advantages/targets or using reward scaling could stabilize learning.

---

## Recommendations (what to try next)

1. **Make off-policy methods truly discrete.**
   Implement **SAC-Discrete** (categorical policy; soft Q over 2 actions) and a **discrete DQN-style baseline** with n-step/λ-returns. This removes the thresholding hack and should materially help.

2. **Strengthen exploration:**

   * **Count-based / RND / ICM** bonuses for PPO and SAC.
   * **ε floor & slower decay** for PQN; keep exploration alive longer on Medium/Hard.
   * **More envs in parallel** so rare successes appear earlier (helps PQN and off-policy).

3. **Budget & curriculum:**
   Increase **total env steps** for Medium/Hard and/or use **curriculum** (train Easy → initialize Medium, etc.). PQN, in particular, should benefit: once it sees the terminal, it learns fast.

4. **Diagnostics to add:**
   Log **success rate** (goal reached), **average action-right ratio** early in episodes, and **buffer hit counts** for terminal transitions. These directly test the “discovery vs propagation” hypothesis.

5. **Tune critical hparams:**

   * **SAC:** auto-tuning α, larger target entropy, slightly larger actor std init.
   * **DDPG:** OU noise with higher σ and slower decay; delayed policy updates.
   * **PPO:** higher entropy bonus; occasional ε-greedy overlay during data collection.
   * **PQN:** try λ ∈ {0.7, 0.9, 0.95}, larger batch, gradient clipping on targets, slower ε decay.

---

## Final narrative

* On **Easy**, once the terminal reward is discovered, **Q(λ)** (PQN) and **entropy-driven exploration** (SAC) rapidly produce near-optimal greedy policies: **proof that the learners and value propagation work**.
* On **Medium** and **Hard**, the **probability of discovery** becomes the dominant factor; without extra exploration mechanisms or longer budgets, **all methods underperform**, with **PPO** slightly ahead on Medium due to stable on-policy updates around the small-reward path.
* These outcomes are **fully consistent** with prior work: PPO/DDPG/SAC are not designed to crack severe sparse exploration on their own; PQN accelerates learning **after** discovery but doesn’t guarantee discovery.

**Main lesson:** ChainEnv at higher difficulty is an **exploration benchmark** first and foremost. Improving **how** we explore (discrete SAC, better ε-schedules, intrinsic bonuses, parallelism, curriculum) should move Medium/Hard from “rare spikes” to **reliable learning**, at which point PQN and SAC should again separate themselves by **how fast** they propagate the sparse signal.
