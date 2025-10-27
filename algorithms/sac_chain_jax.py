# algorithms/sac_chain_jax.py
# ==========================================================
# SAC baseline (continuous ChainEnv version)
# Logs episodic returns to runs/sac/<difficulty>.csv
# Based off of jaxrl2 sac implementation
# ==========================================================
import os
import numpy as np
import jax
import jax.numpy as jnp
from external.jaxrl2_sac.sac_learner import SACLearner
from envs.chain_jax_env import reset, step, DIFFICULTIES
from algorithms.utils import get_run_config

def collect_returns_sac(learner, env_params, num_envs, seed, max_env_steps):
    rng = jax.random.PRNGKey(seed + 123)
    reset_keys = jax.random.split(rng, num_envs)
    env_state, obs = jax.vmap(reset, in_axes=(0, None))(reset_keys, env_params)
    obs = obs.reshape(num_envs, -1)

    ep_ret = jnp.zeros(num_envs)
    ep_returns, ep_steps = [], []
    steps = 0

    while steps < max_env_steps:
        actions = learner.eval_actions(obs)  # deterministic tanh(mu)
        actions = jnp.array(actions).reshape(num_envs, -1)

        next_state, next_obs, reward, done = jax.vmap(step, in_axes=(0, 0, None))(
            env_state, actions[:, 0], env_params
        )

        ep_ret = ep_ret + reward
        steps += num_envs

        if np.any(np.asarray(done)):
            finished = np.asarray(ep_ret[np.where(np.asarray(done))])
            ep_returns.append(float(np.mean(finished)))
            ep_steps.append(steps)
            ep_ret = jnp.where(jnp.array(done), 0.0, ep_ret)

            rng, subk = jax.random.split(rng)
            count = int(np.asarray(done).sum())
            new_keys = jax.random.split(subk, count)
            new_states, new_obs = jax.vmap(reset, in_axes=(0, None))(new_keys, env_params)

            s, t, k = map(np.array, (next_state.s, next_state.t, next_state.key))
            ob = np.array(next_obs)
            idx = np.where(np.asarray(done))[0]
            s[idx] = np.array(new_states.s); t[idx] = np.array(new_states.t); k[idx] = np.array(new_states.key)
            ob[idx] = np.array(new_obs)

            env_state = next_state._replace(s=jnp.array(s), t=jnp.array(t), key=jnp.array(k))
            obs = jnp.array(ob)
        else:
            env_state, obs = next_state, next_obs

    return np.column_stack([np.array(ep_steps, dtype=np.int64),
                            np.array(ep_returns, dtype=np.float32)])

def main(config):
    difficulty = config["DIFFICULTY"]
    env_params = DIFFICULTIES[difficulty]

    rng = jax.random.PRNGKey(config["SEED"])
    num_envs = config["NUM_ENVS"]
    total_env_steps = config["TOTAL_ENV_STEPS"]

    reset_keys = jax.random.split(rng, num_envs)
    env_state, obs = jax.vmap(reset, in_axes=(0, None))(reset_keys, env_params)
    obs = obs.reshape(num_envs, -1)  # (B, 1)

    obs_dim = obs.shape[-1]
    act_dim = 1
    dummy_obs = jnp.zeros((obs_dim,))
    dummy_act = jnp.zeros((act_dim,))
    learner = SACLearner(
        seed=config["SEED"],
        observations=dummy_obs,
        actions=dummy_act,
        actor_lr=config["ACTOR_LR"],
        critic_lr=config["CRITIC_LR"],
        hidden_dims=(64, 64),
        discount=config["GAMMA"],
        tau=config["TAU"],
        alpha=config["ALPHA"],
    )

    buffer, buffer_limit = [], config["BUFFER_SIZE"]
    ep_ret = jnp.zeros(num_envs)
    ep_returns, ep_steps = [], []
    global_step = 0

    print(f"Running SAC | Difficulty={difficulty} | Steps={total_env_steps}")

    while global_step < total_env_steps:
        # batch-safe: one call, no vmap over a stateful method
        actions = learner.sample_actions(obs)  # (B, 1)

        next_state, next_obs, reward, done = jax.vmap(step, in_axes=(0, 0, None))(
            env_state, actions[:, 0], env_params
        )

        reward = np.asarray(reward, dtype=np.float32)
        next_obs = np.asarray(next_obs, dtype=np.float32).reshape(num_envs, -1)
        done = np.asarray(done, dtype=bool)

        # store transitions
        for i in range(num_envs):
            buffer.append(
                (
                    np.asarray(obs[i]),
                    np.asarray(actions[i]),
                    np.asarray(reward[i]),
                    np.asarray(next_obs[i]),
                    np.asarray(done[i]),
                )
            )
        if len(buffer) > buffer_limit:
            buffer = buffer[-buffer_limit:]

        # update
        if len(buffer) >= config["BATCH_SIZE"]:
            batch_idx = np.random.choice(len(buffer), config["BATCH_SIZE"], replace=False)
            b_obs, b_act, b_rew, b_next_obs, b_done = map(np.stack, zip(*[buffer[i] for i in batch_idx]))
            batch_jax = {
                "observations": jnp.array(b_obs),
                "actions": jnp.array(b_act).reshape(-1, 1),
                "rewards": jnp.array(b_rew).reshape(-1, 1),
                "next_observations": jnp.array(b_next_obs),
                "masks": 1.0 - jnp.array(b_done, dtype=jnp.float32).reshape(-1, 1),
            }
            learner.update(batch_jax)

        # episodic logging
        ep_ret += reward
        global_step += num_envs
        if jnp.any(done):
            finished_returns = np.asarray(ep_ret[np.where(np.asarray(done))])
            ep_returns.append(float(np.mean(finished_returns)))
            ep_steps.append(global_step)
            ep_ret = jnp.where(jnp.array(done), 0.0, ep_ret)


        # resets
        reset_mask = np.array(done)
        if reset_mask.any():
            rng, subk = jax.random.split(rng)
            new_keys = jax.random.split(subk, int(reset_mask.sum()))
            new_states, new_obs = jax.vmap(reset, in_axes=(0, None))(new_keys, env_params)
            s, t, k = map(np.array, (env_state.s, env_state.t, env_state.key))
            obs_arr = np.array(next_obs)
            idx = np.where(reset_mask)[0]
            s[idx] = np.array(new_states.s)
            t[idx] = np.array(new_states.t)
            k[idx] = np.array(new_states.key)
            obs_arr[idx] = np.array(new_obs)
            env_state = env_state._replace(s=jnp.array(s), t=jnp.array(t), key=jnp.array(k))
            obs = jnp.array(obs_arr)
        else:
            env_state, obs = next_state, next_obs

    os.makedirs("runs/sac", exist_ok=True)
    out_path = f"runs/sac/{difficulty}.csv"
    eval_curves = collect_returns_sac(learner, env_params, num_envs, config["SEED"], total_env_steps)
    eval_path = f"runs/sac/{difficulty}_eval.csv"
    np.savetxt(eval_path, eval_curves, fmt=["%d", "%.6f"], delimiter=",")
    print(f"Saved deterministic eval returns to {eval_path}")

    np.savetxt(out_path, np.column_stack([ep_steps, ep_returns]), fmt=["%d", "%.6f"], delimiter=",")
    print(f"Saved episodic returns to {out_path}")


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
        "ALPHA": 0.2,
    }
    main(config)
