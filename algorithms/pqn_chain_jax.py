# algorithms/pqn_chain_jax.py
# ==========================================================
# PureJaxQL-style PQN on ChainEnv (JAX)
# - On-policy rollouts
# - Q(λ) targets computed via reverse scan
# - ε scheduled by update index
# - RAdam + grad-clip in learner
# ==========================================================
import os
import numpy as np
import jax
import jax.numpy as jnp
import optax

from envs.chain_jax_env import batch_reset, batch_step, DIFFICULTIES
from external.purejaxql_pqn.pqn_learner import PQNLearner

ALGO_NAME = "pqn"

# ---------------- helpers ----------------
def make_eps_schedule(total_timesteps_decay, num_envs, num_steps,
                      eps_start=1.0, eps_finish=0.05, eps_decay_ratio=0.2):
    # PureJaxQL: decay is in *updates*, not raw env steps
    num_updates = int(total_timesteps_decay // (num_envs * num_steps))
    transition_steps = max(1, int(eps_decay_ratio * num_updates))
    return num_updates, optax.linear_schedule(eps_start, eps_finish, transition_steps)

def lambda_targets(q_seq, rew, done, last_q, gamma, lam):
    """
    PJQL-style Q(λ) over time-major rollout.
      q_seq: (T,B,A)     rew,done: (T,B)
      last_q: (B,) = max_a Q(s_T, a)
    returns: (T,B) λ-returns
    """
    # start from t = T-1 using next_q = last_q
    lamret_Tm1 = rew[-1] + gamma * (1.0 - done[-1]) * last_q  # (B,)

    def scan_back(lamret_next, inputs):
        r_t, d_t, q_t = inputs           # each (B,*) at time t
        qmax_t = jnp.max(q_t, axis=-1)   # (B,)
        boot = r_t + gamma * (1.0 - d_t) * qmax_t
        delta = lamret_next - qmax_t
        lamret = boot + gamma * lam * delta
        lamret = (1.0 - d_t) * lamret + d_t * r_t  # cut on terminal
        return lamret, lamret

    _, lamrets_rev = jax.lax.scan(
        scan_back,
        lamret_Tm1,
        (rew[:-1][::-1], done[:-1][::-1], q_seq[:-1][::-1]),
    )
    lamrets = jnp.concatenate([lamrets_rev[::-1], lamret_Tm1[None]], axis=0)
    return lamrets  # (T,B)
# ---------------- deterministic evaluation ----------------
def collect_returns_greedy(learner, env_params, seed, max_env_steps=80000, tie_bias=1e-6):
    """Evaluate the trained PQN policy deterministically (no ε exploration).

    tie_bias: add a tiny constant to Q[:,1] so exact ties choose action=1 (move right).
              Set to 0.0 to disable and use strict argmax.
    """
    rng = jax.random.PRNGKey(seed + 123)
    NUM_ENVS = 8
    keys = jax.random.split(rng, NUM_ENVS)
    env_states, obs = batch_reset(keys, env_params)

    ep_ret = jnp.zeros(NUM_ENVS)
    ep_returns, ep_steps = [], []
    steps = 0

    while steps < max_env_steps:
        q_vals = learner.q.apply_fn({"params": learner.q.params}, obs)  # (B,2)
        if tie_bias:
            q_vals = q_vals.at[:, 1].add(jnp.asarray(tie_bias, dtype=q_vals.dtype))
        acts = jnp.argmax(q_vals, axis=-1)  # {0,1}; env maps >0 to +1

        next_states, next_obs, rew, done = batch_step(env_states, acts, env_params)
        ep_ret = ep_ret + rew
        steps += NUM_ENVS

        if bool(jnp.any(done)):
            finished = np.asarray(ep_ret[np.where(np.asarray(done))])
            ep_returns.append(float(np.mean(finished)))
            ep_steps.append(steps)

            reset_count = int(np.asarray(done).sum())
            rng, subk = jax.random.split(rng)
            new_keys = jax.random.split(subk, reset_count)
            new_states, new_obs = batch_reset(new_keys, env_params)

            s = np.array(next_states.s); t = np.array(next_states.t); k = np.array(next_states.key)
            ob = np.array(next_obs); er = np.array(ep_ret)
            idx = np.where(np.asarray(done))[0]
            s[idx] = np.array(new_states.s); t[idx] = np.array(new_states.t)
            k[idx] = np.array(new_states.key); ob[idx] = np.array(new_obs)
            er[idx] = 0.0

            env_states = env_states._replace(s=jnp.array(s), t=jnp.array(t), key=jnp.array(k))
            obs = jnp.array(ob); ep_ret = jnp.array(er)
        else:
            env_states, obs = next_states, next_obs

    return np.column_stack([
        np.array(ep_steps, dtype=np.int64),
        np.array(ep_returns, dtype=np.float32)
    ])

# ---------------- main ----------------
def run_pqn(env_params, total_steps=10000, seed=0):
    rng = jax.random.PRNGKey(seed)

    # Training config (PJQL-ish; small for Chain)
    NUM_ENVS = 8
    NUM_STEPS = 32
    NUM_MINIBATCHES = 4
    NUM_EPOCHS = 2
    GAMMA = 0.99
    LAMBDA = 0.9
    LR = 2.5e-4
    MAX_GRAD_NORM = 10.0

    # ε schedule by update index
    num_updates_all, eps_sched = make_eps_schedule(
        total_timesteps_decay=float(total_steps),
        num_envs=NUM_ENVS,
        num_steps=NUM_STEPS,
        eps_start=1.0, eps_finish=0.05, eps_decay_ratio=0.2,
    )
    updates_to_do = max(1, total_steps // (NUM_ENVS * NUM_STEPS))

    learner = PQNLearner(seed, obs_dim=1, n_actions=2,
                         hidden_dims=(128, 128), lr=LR, max_grad_norm=MAX_GRAD_NORM)

    # init envs
    rng, env_key = jax.random.split(rng)
    keys = jax.random.split(env_key, NUM_ENVS)
    env_states, obs = batch_reset(keys, env_params)          # obs: (B, 1)

    ep_ret = jnp.zeros(NUM_ENVS)
    ep_returns, ep_steps = [], []
    global_steps = 0

    for update_idx in range(updates_to_do):
        eps = float(eps_sched(update_idx))

        # ------- collect rollout (time-major) -------
        obs_seq, act_seq, rew_seq, done_seq, q_seq = [], [], [], [], []
        for _ in range(NUM_STEPS):
            # compute q(s) once per step for λ-targets
            q_vals = learner.q.apply_fn({"params": learner.q.params}, obs)   # (B, A)
            rng, a_key, e_key = jax.random.split(rng, 3)
            greedy  = jnp.argmax(q_vals, axis=-1)
            rand    = jax.random.randint(a_key, greedy.shape, 0, learner.n_actions) # 2 actions in Chain
            explore = jax.random.uniform(e_key, greedy.shape) < eps
            acts    = jnp.where(explore, rand, greedy)

            next_states, next_obs, rew, done = batch_step(env_states, acts, env_params)

            obs_seq.append(obs)
            act_seq.append(acts)
            rew_seq.append(rew)
            done_seq.append(done.astype(jnp.float32))
            q_seq.append(q_vals)

            # episodic logging/reset (kept from your code)
            ep_ret = ep_ret + rew
            global_steps += NUM_ENVS
            if bool(jnp.any(done)):
                finished = np.asarray(ep_ret[np.where(np.asarray(done))])
                ep_returns.append(float(np.mean(finished)))
                ep_steps.append(global_steps)
                # reset finished envs
                reset_count = int(np.asarray(done).sum())
                rng, subk = jax.random.split(rng)
                new_keys = jax.random.split(subk, reset_count)
                new_states, new_obs = batch_reset(new_keys, env_params)

                s = np.array(next_states.s); tcounter = np.array(next_states.t); k = np.array(next_states.key)
                ob = np.array(next_obs); er = np.array(ep_ret)
                idx = np.where(np.asarray(done))[0]
                s[idx] = np.array(new_states.s); tcounter[idx] = np.array(new_states.t)
                k[idx] = np.array(new_states.key); ob[idx] = np.array(new_obs)
                er[idx] = 0.0
                env_states = env_states._replace(s=jnp.array(s), t=jnp.array(tcounter), key=jnp.array(k))
                obs = jnp.array(ob); ep_ret = jnp.array(er)
            else:
                env_states, obs = next_states, next_obs

        # stack time-major
        obs_seq  = jnp.stack(obs_seq)            # (T,B,1)
        act_seq  = jnp.stack(act_seq)            # (T,B)
        rew_seq  = jnp.stack(rew_seq)            # (T,B)
        done_seq = jnp.stack(done_seq)           # (T,B)
        q_seq    = jnp.stack(q_seq)              # (T,B,A)

        # compute last_q = max_a Q(s_T)
        last_q_vals = learner.q.apply_fn({"params": learner.q.params}, obs)
        last_q = jnp.max(last_q_vals, axis=-1)   # (B,)

        # λ-returns
        lam_tgts = lambda_targets(q_seq, rew_seq, done_seq, last_q, GAMMA, LAMBDA)  # (T,B)

        # ------- flatten, shuffle, train -------
        T, B = lam_tgts.shape
        flat_obs = obs_seq.reshape(T * B, -1)     # (TB, obs_dim)
        flat_act = act_seq.reshape(T * B)         # (TB,)
        flat_tgt = lam_tgts.reshape(T * B)        # (TB,)

        rng, perm_key = jax.random.split(rng)
        perm = jax.random.permutation(perm_key, T * B)
        flat_obs, flat_act, flat_tgt = flat_obs[perm], flat_act[perm], flat_tgt[perm]

        mb = (T * B) // NUM_MINIBATCHES
        for _ in range(NUM_EPOCHS):
            for i in range(NUM_MINIBATCHES):
                sl = slice(i * mb, (i + 1) * mb)
                learner.train_minibatch(flat_obs[sl], flat_act[sl], flat_tgt[sl])

    # return episodic returns curve and the trained learner
    train_curve = np.column_stack([
        np.array(ep_steps, dtype=np.int64),
        np.array(ep_returns, dtype=np.float32)
    ])
    return train_curve, learner


if __name__ == "__main__":
    difficulty = os.getenv("CHAIN_DIFFICULTY", "medium")
    total_steps = int(float(os.getenv("CHAIN_TOTAL_ENV_STEPS", "80000")))
    seed = int(os.getenv("CHAIN_SEED", 0))

    env_params = DIFFICULTIES[difficulty]
    print(f" Running PQN (PureJaxQL-style) | Difficulty={difficulty} | Steps={total_steps}")
        
    # --- Train ---
    train_curve, learner = run_pqn(env_params, total_steps, seed)

    # Save training curve (ε-greedy)
    os.makedirs(f"runs/{ALGO_NAME}", exist_ok=True)
    out_train = f"runs/{ALGO_NAME}/{difficulty}.csv"
    np.savetxt(out_train, train_curve, fmt=["%d","%.6f"], delimiter=",")
    print(f" Saved training (ε-greedy) returns to {out_train}")

    # --- Deterministic evaluation using the **trained** learner ---
    print(" Running deterministic PQN evaluation...")
    eval_returns = collect_returns_greedy(
        learner=learner,
        env_params=env_params,
        seed=seed,
        max_env_steps=total_steps,
        tie_bias=1e-6,  # set to 0.0 to disable the tie-break
    )
    out_eval = f"runs/{ALGO_NAME}/{difficulty}_eval.csv"
    np.savetxt(out_eval, eval_returns, fmt=["%d","%.6f"], delimiter=",")
    print(f" Saved deterministic evaluation returns to {out_eval}")
