# algorithms/ddpg_chain_jax.py
"""
DDPG baseline (continuous ChainEnv version)
------------------------------------------
- Trains a deterministic policy with Gaussian exploration on a 1-D ChainEnv.
- Uses a *simple Python-list replay buffer* for clarity (each step appends tuples).
- Vectorized environments via `vmap` (NUM_ENVS parallel copies).
- Logs 2 CSVs per difficulty:
* `runs/ddpg/<difficulty>.csv` — training episodic returns (with noise)
* `runs/ddpg/<difficulty>_eval.csv` — deterministic evaluation (no noise)


Key shapes:
- obs: (B, 1) — chain position as float
- act: (B, 1) — in [-1, 1]; we feed the scalar to env as left/right
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
from external.jaxrl_ddpg import DDPGLearner
from envs.chain_jax_env import reset, step, EnvParams, DIFFICULTIES
from algorithms.utils import get_run_config

def collect_returns_ddpg(learner, env_params, num_envs, seed, max_env_steps):
    
    """Deterministic evaluation loop (no exploration noise).
   
     We call the actor directly (no sampling noise) and roll forward `num_envs`
    environments in parallel. Whenever at least one env finishes, we log the
    mean return among the finished envs and immediately reset only those envs.
    """

    rng = jax.random.PRNGKey(seed + 123)
    reset_keys = jax.random.split(rng, num_envs)
    
    # Vectorized reset → parallel state & observation batch
    env_state, obs = jax.vmap(reset, in_axes=(0, None))(reset_keys, env_params)
    obs = obs.reshape(num_envs, -1)

    ep_ret = jnp.zeros(num_envs)
    ep_returns, ep_steps = [], []
    steps = 0

    while steps < max_env_steps:
        # Greedy action from the current policy (no noise in eval)
        actions = learner.actor_apply(learner.actor_params, obs)
        actions = jnp.array(actions).reshape(num_envs, -1)

        # One parallel environment step
        next_state, next_obs, reward, done = jax.vmap(step, in_axes=(0, 0, None))(
            env_state, actions[:, 0], env_params
        )

        ep_ret = ep_ret + reward
        steps += num_envs

        # If any env finished, log mean return of the finished subset and reset them
        if np.any(np.asarray(done)):
            finished = np.asarray(ep_ret[np.where(np.asarray(done))])
            ep_returns.append(float(np.mean(finished)))
            ep_steps.append(steps)
            
            # Zero only the finished environments' episodic accumulators
            ep_ret = jnp.where(jnp.array(done), 0.0, ep_ret)

            # Reset finished envs in-place, keep ongoing ones untouched
            rng, subk = jax.random.split(rng)
            count = int(np.asarray(done).sum())
            new_keys = jax.random.split(subk, count)
            new_states, new_obs = jax.vmap(reset, in_axes=(0, None))(new_keys, env_params)

            # In-place selective swap for finished indices
            s, t, k = map(np.array, (next_state.s, next_state.t, next_state.key))
            ob = np.array(next_obs)
            idx = np.where(np.asarray(done))[0]
            s[idx] = np.array(new_states.s); t[idx] = np.array(new_states.t); k[idx] = np.array(new_states.key)
            ob[idx] = np.array(new_obs)

            env_state = next_state._replace(s=jnp.array(s), t=jnp.array(t), key=jnp.array(k))
            obs = jnp.array(ob)
        else:
            env_state, obs = next_state, next_obs

    # Return a 2-col array: [env_steps, episodic_return]
    return np.column_stack([np.array(ep_steps, dtype=np.int64),
                            np.array(ep_returns, dtype=np.float32)])

def main(config):
    difficulty = config["DIFFICULTY"]
    env_params = DIFFICULTIES[difficulty]

    rng = jax.random.PRNGKey(config["SEED"])
    num_envs = config["NUM_ENVS"]
    total_env_steps = config["TOTAL_ENV_STEPS"]

    # -----------------------------
    # Initialize vectorized envs
    # -----------------------------
    reset_keys = jax.random.split(rng, num_envs)
    env_state, obs = jax.vmap(reset, in_axes=(0, None))(reset_keys, env_params)
    obs = obs.reshape(num_envs, -1)
    # -----------------------------
    # Initialize learner
    # -----------------------------
    # We pass *dummy* shapes to initialize networks (pure JAX MLPs).
    obs_dim = obs.shape[-1]
    act_dim = 1  # continuous scalar actions
    dummy_obs = jnp.zeros((obs_dim,))
    dummy_act = jnp.zeros((act_dim,))
    learner = DDPGLearner(
        seed=config["SEED"],
        observations=dummy_obs,
        actions=dummy_act,
        actor_lr=config["ACTOR_LR"],
        critic_lr=config["CRITIC_LR"],
        hidden_dims=(64, 64),
        discount=config["GAMMA"],
        tau=config["TAU"],
        target_update_period=1,
        exploration_noise=config["EXPL_NOISE"],
    )

    # -----------------------------
    # Minimal replay buffer (list of tuples)
    # -----------------------------
    # Each entry: (obs, act, rew, next_obs, done)
    # This is intentionally simple for readability; you can swap in the
    # JAX ring buffer from algorithms/jax_buffer.py later without changing
    # the learner API.
    buffer = []
    buffer_limit = config["BUFFER_SIZE"]


    # -----------------------------
    # Logging
    # -----------------------------
    ep_ret = jnp.zeros(num_envs)
    ep_returns, ep_steps = [], []
    global_step = 0

    print(f"Running DDPG | Difficulty={difficulty} | Steps={total_env_steps}")



    # -----------------------------
    # Main training loop (interleave collect + learn)
    # -----------------------------
    while global_step < total_env_steps:
        # 1) Collect: sample *noisy* actions for exploration
        actions = learner.sample_actions(obs)
        actions = jnp.array(actions).reshape(num_envs, -1)

        # 2) Step all envs in parallel
        single_keys = env_state.key  # shape (num_envs, 2)
        env_state_unbatched = env_state._replace(key=single_keys)

        # Convert to NumPy once for easy Python-list appends
        next_state, next_obs, reward, done = jax.vmap(step, in_axes=(0, 0, None))(
            env_state_unbatched, actions[:, 0], env_params
        )

        reward = np.asarray(reward, dtype=np.float32)
        next_obs = np.asarray(next_obs, dtype=np.float32).reshape(num_envs, -1)
        done = np.asarray(done, dtype=bool)

        # 3) Store transitions in the replay buffer
        for i in range(num_envs):
            buffer.append(
                (
                    np.asarray(obs[i], dtype=np.float32).reshape(-1),
                    np.asarray(actions[i], dtype=np.float32).reshape(-1),
                    np.asarray(reward[i], dtype=np.float32).reshape(()),
                    np.asarray(next_obs[i], dtype=np.float32).reshape(-1),
                    np.asarray(done[i], dtype=bool).reshape(()),
                )
            )
        # 4) Update the learner (off-policy minibatch)
        if len(buffer) > buffer_limit:
            buffer = buffer[-buffer_limit:]

        # Update learner (off-policy)
        if len(buffer) >= config["BATCH_SIZE"]:
            batch_idx = np.random.choice(len(buffer), config["BATCH_SIZE"], replace=False)
            batch = [buffer[i] for i in batch_idx]
            b_obs, b_act, b_rew, b_next_obs, b_done = map(np.stack, zip(*batch))
            # Learner expects masks = 1.0 for nonterminal transitions
            batch_jax = {
                "observations": jnp.array(b_obs),
                "actions": jnp.array(b_act).reshape(-1, 1),
                "rewards": jnp.array(b_rew).reshape(-1, 1),
                "next_observations": jnp.array(b_next_obs),
                "masks": 1.0 - jnp.array(b_done, dtype=jnp.float32).reshape(-1, 1),
            }
            learner.update(batch_jax)

        # 5) Episodic logging (tracks mean return of any finished envs)
        ep_ret = ep_ret + reward
        global_step += num_envs

        if jnp.any(done):
            finished_returns = np.asarray(ep_ret[np.where(np.asarray(done))])
            ep_returns.append(float(np.mean(finished_returns)))
            ep_steps.append(global_step)

            # zero-out finished env accumulators; ongoing envs keep their returns
            ep_ret = jnp.where(jnp.array(done), 0.0, ep_ret)


        # 6) Selectively reset finished envs
        reset_mask = np.array(done, dtype=bool)
        if reset_mask.any():
            rng, subk = jax.random.split(rng)
            num_resets = int(reset_mask.sum())
            new_keys = jax.random.split(subk, num_resets)
            new_states, new_obs = jax.vmap(reset, in_axes=(0, None))(new_keys, env_params)

            s = np.array(env_state.s)
            t = np.array(env_state.t)
            key = np.array(env_state.key)
            obs_arr = np.array(next_obs)

            reset_indices = np.where(reset_mask)[0]
            s[reset_indices] = np.array(new_states.s)
            t[reset_indices] = np.array(new_states.t)
            key[reset_indices] = np.array(new_states.key)
            obs_arr[reset_indices] = np.array(new_obs)

            env_state = env_state._replace(s=jnp.array(s), t=jnp.array(t), key=jnp.array(key))
            obs = jnp.array(obs_arr)
        else:
            env_state = next_state
            obs = next_obs
        
    # -----------------------------
    # Save curves (training + eval)
    # -----------------------------
    os.makedirs("runs/ddpg", exist_ok=True)
    
    # Training curve (with exploration noise)
    out_path = f"runs/ddpg/{difficulty}.csv"
    out = np.column_stack([ep_steps, ep_returns])
    np.savetxt(out_path, out, fmt=["%d", "%.6f"], delimiter=",")
    print(f"Saved episodic returns to {out_path}")

    # Deterministic evaluation curve (greedy policy)
    eval_curves = collect_returns_ddpg(learner, env_params, num_envs, config["SEED"], total_env_steps)
    eval_path = f"runs/ddpg/{difficulty}_eval.csv"
    np.savetxt(eval_path, eval_curves, fmt=["%d", "%.6f"], delimiter=",")
    print(f"Saved deterministic eval returns to {eval_path}")


if __name__ == "__main__":
    difficulty, total_steps, seed = get_run_config()
    config = {
        "DIFFICULTY": difficulty,
        "TOTAL_ENV_STEPS": total_steps,
        "SEED": seed,
        "NUM_ENVS": 8,
        "BUFFER_SIZE": 10000,
        "BATCH_SIZE": 64,
        "GAMMA": 0.99,
        "TAU": 0.005,
        "ACTOR_LR": 3e-4,
        "CRITIC_LR": 3e-4,
        "EXPL_NOISE": 0.1, # std for Gaussian noise in sample_actions
    }
    main(config)
