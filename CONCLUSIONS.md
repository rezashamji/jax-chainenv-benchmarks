# CONCLUSIONS.md 

## Figures referenced

The following plots visualize the summarized results discussed below.  
All figures are located in the `runs/` folder of this repository.

| Figure | Description |
|:--|:--|
| **chainenv_train_by_algorithm.png** | Training curves (with exploration noise) grouped **by algorithm** across all difficulty levels. |
| **chainenv_train_by_difficulty.png** | Training curves (with exploration noise) grouped **by difficulty** across all algorithms. |
| **chainenv_eval_by_algorithm.png** | Deterministic **evaluation (greedy)** grouped **by algorithm**. |
| **chainenv_eval_by_difficulty.png** | Deterministic **evaluation (greedy)** grouped **by difficulty**. |

These correspond respectively to the **training** (exploratory) and **evaluation** (greedy) phases described below.

---

## What the results show (at a glance)

**Max greedy return per preset (theoretical upper bound if you just go straight to the goal):**
- **Easy:** `r_small + r_big = 0.3 + 1.0 = 1.3`
- **Medium:** `0.1 + 1.0 = 1.1`
- **Hard:** `0.0 + 1.0 = 1.0`

**Easy chain:**
- **SAC** and **PQN** reliably reach the **final (big) reward**; greedy evaluation is often ~2.3–2.5 because policies frequently collect the small reward at position 1 several times before finishing; the straight-to-goal baseline is 1.3.`
- **PPO** and **DDPG** often hover around the **local lure** and/or fall short of consistently reaching the goal within budget.

**Medium chain:**
- No algorithm consistently solves the task.
- **PPO** is typically the highest (~**near the small-reward + occasional progress**), with **DDPG close behind** and **SAC ≈ PQN** tend to settle near suboptimal returns.

**Hard chain:**
- **None** of the algorithms reliably solve it within the current budgets.
- **PPO and DDPG** intermittently reach near-goal returns but lack stability; averages remain below the maximum.
- **SAC** is mostly near zero with rare blips. PQN shows rare spikes (discoveries) without stable conversion.

These patterns are consistent across **training** (noisy policy sampling for PPO, Gaussian/tanh-Gaussian for DDPG/SAC, ε-greedy for PQN) and **eval** (greedy) curves. Eval removes exploration noise and shows the true learned policy.

---

## Per-algorithm takeaways

### PPO (on-policy, clipped PG with GAE)
- **Easy:** tends to get pulled to the **local optimum**; without explicit exploration bonuses it struggles to consistently discover the long rightward path.
- **Medium/Hard:** mostly stuck; slight advantage on **Medium** likely from stable on-policy updates around the small-reward path.
- **Why:** on-policy sampling doesn’t frequently reach the sparse terminal reward; clipping+GAE stabilize optimization but don’t create exploration.

### DDPG (off-policy, deterministic actor + twin Q + targets)
- **Easy:** often tracks small reward; discovery of the long right path is unreliable.
- **Medium/Hard:** fails to explore; replay rarely contains successful terminal transitions.
- **Why:** deterministic policy + simple Gaussian noise is **exploration-poor** on sparse, long-horizon tasks.

### SAC (off-policy, stochastic actor with entropy)
- **Easy:** commonly **solves**; entropy keeps exploration broad enough to hit the goal early, then twin-Q/targets consolidate it.
- **Medium/Hard:** entropy alone is **insufficient** here; the chance of stumbling into the full rightward trajectory remains low within budget.
- **Why:** better exploration than DDPG but still fundamentally chance-based without intrinsic bonuses/curriculum.

### PQN (on-policy Q-learning with **Q(λ)**)
- **Easy:** **solves cleanly** in greedy evaluation; training can show dips because ε-greedy exploration persists during training.
- **Medium:** plateaus suboptimally; **λ-returns** help **after** a success but do not make successes more frequent.
- **Hard:** **occasional spikes** (some discoveries) without stable conversion to policy improvement before ε decays or budget ends.
- **Why:** PQN’s strength is **fast credit propagation once success happens**; bottleneck is **discovery**, not propagation.

---

## Why this is happening (mechanistic read)

1. **Exploration barrier dominates.**  
   The agent must string together many **right moves** and resist the **small-reward lure** at position 1. With longer horizons and **slip**, the probability of ever seeing the terminal reward collapses.

2. **Small-reward harvesting on Easy.**
    With H=15 and r_small=0.3, a policy can oscillate between 0 and 1 multiple times and still reach the goal, so total return can exceed the straight-to-goal 1.3.`

3. **On- vs off-policy differences.**  
   - **On-policy** (PPO, PQN): explore with the **current** behavior; if that behavior locks on the small reward, learning saturates.  
   - **Off-policy** (DDPG, SAC): can in principle learn from **rare successes** if they land in replay, but without them the targets stay flat.

4. **Why SAC > DDPG on Easy.**  
   SAC’s entropy sustains **wider action coverage**; once a few successes enter replay, twin-Q + targets stabilize the better policy. DDPG’s exploration is narrower.

5. **Why PQN shines on Easy but not Medium/Hard.**  
   **Q(λ)** propagates reward **quickly** once observed; it doesn’t increase the odds of the **first** success.

6. **Training vs Eval (algorithm-specific).**  
   Training curves include exploration noise: **PPO** samples from a categorical, **DDPG** uses Gaussian noise, **SAC** samples tanh-Gaussian actions, **PQN** is ε-greedy. Greedy **eval** removes that noise and shows the actual policy.

---

## Alignment with the papers

- **PPO (Schulman et al., 2017):** Clipped policy ratios + GAE stabilize on-policy learning; no built-in sparse-exploration fix. **Observed behavior aligns.**
- **DDPG (Lillicrap et al., 2015/2016):** Strong on dense control; brittle exploration on sparse tasks. **Observed behavior aligns.** (This implementation uses Gaussian rather than the paper’s OU noise. This doesn’t change the core point.)
- **SAC (Haarnoja et al., 2018):** Maximum-entropy objective improves exploration but doesn’t guarantee solving long-horizon sparse tasks. **Observed behavior aligns.**
- **PQN / Q(λ):** λ-returns accelerate credit assignment **after** success; discovery remains the bottleneck. **Observed behavior aligns.**

**Bottom line:** performance is **exploration-limited**, not optimizer-limited, as difficulty grows.

---

## Caveats & likely contributors

- **Action-space mismatch for DDPG/SAC.** Policies are continuous but the env interprets **sign(action)** as {left,right}. This removes gradient structure for direction choice; **SAC-Discrete** / discrete critics may help.
- **Hyperparameters are generic.** α (SAC), noise scale (DDPG), entropy bonus (PPO), ε-schedule/λ (PQN) are not tuned for severe sparsity.
- **Single-seed sensitivity.** Multi-seed runs (e.g., 10 seeds, mean ± 95% CI) would be more faithful.
- **Reward scale vs targets.** Advantage/value normalization and/or reward scaling can matter.

---

## Recommendations

1. **Make off-policy methods discrete.** Try **SAC-Discrete** (categorical policy, soft Q over 2 actions) and a DQN-style baseline with n-step/λ-returns.
2. **Strengthen exploration:** count/RND/ICM bonuses; slower ε decay or ε floor for PQN; more parallel envs.
3. **Budget & curriculum:** expand **Medium/Hard** budgets; or train **Easy → initialize Medium**, etc.
4. **Diagnostics:** log **success rate**, **right-action ratio** in early steps, and replay **terminal-transition counts**.
5. **Hyperparam sweeps:**
   - **SAC:** autotune α, larger target entropy, higher init std.
   - **DDPG:** stronger/decaying noise (OU or larger Gaussian), delayed policy updates.
   - **PPO:** higher entropy bonus; optional ε-greedy overlay during collection.
   - **PQN:** λ ∈ {0.7, 0.9, 0.95}, larger batch, grad-clip, slower ε decay.

---

## Final narrative

- On **Easy**, once the terminal reward is discovered, **PQN (Q(λ))** and **SAC (entropy)** quickly produce high greedy returns (~2.3–2.5 on Easy due to small-reward harvesting).  
- On **Medium/Hard**, the **probability of discovery** dominates; all methods underperform without extra exploration or budget, with **PPO** often slightly ahead on Medium.
- This mirrors the literature: PPO/DDPG/SAC don’t *solve* sparse exploration by themselves; PQN excels **after** discovery.

**Main lesson:** ChainEnv at higher difficulty is an **exploration benchmark**. Improve *how we explore* (discrete SAC, intrinsic bonuses, ε-schedules, parallelism, curriculum) to turn Medium/Hard from “rare spikes” into **reliable learning**.
```
---
