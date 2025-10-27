# algorithms/ppo_chain_jax.py
# ==========================================================
# PPO baseline (JAX-native) for ChainEnv exploration benchmark
# Adapted from purejaxrl/ppo_continuous_action.py
# ==========================================================

import os
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.training.train_state import TrainState
from flax.linen.initializers import orthogonal
from typing import NamedTuple
import distrax

from envs.chain_jax_env import reset, step, batch_reset, batch_step, DIFFICULTIES, EnvParams

ALGO_NAME = "ppo"


class ActorCritic(nn.Module):
    action_dim: int = 2  # left or right
    hidden_dim: int = 64
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        act_fn = nn.relu if self.activation == "relu" else nn.tanh

        # Actor
        a = act_fn(nn.Dense(self.hidden_dim, kernel_init=orthogonal(jnp.sqrt(2.0)))(x))
        a = act_fn(nn.Dense(self.hidden_dim, kernel_init=orthogonal(jnp.sqrt(2.0)))(a))
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(a)
        pi = distrax.Categorical(logits=logits)

        # Critic
        v = act_fn(nn.Dense(self.hidden_dim, kernel_init=orthogonal(jnp.sqrt(2.0)))(x))
        v = act_fn(nn.Dense(self.hidden_dim, kernel_init=orthogonal(jnp.sqrt(2.0)))(v))
        value = nn.Dense(1, kernel_init=orthogonal(1.0))(v)

        return pi, jnp.squeeze(value, -1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray


def make_train(config, env_params: EnvParams):
    num_updates = config["TOTAL_ENV_STEPS"] // (config["NUM_STEPS"] * config["NUM_ENVS"])
    minibatch_size = (config["NUM_ENVS"] * config["NUM_STEPS"]) // config["NUM_MINIBATCHES"]

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / num_updates
        return config["LR"] * frac

    def train(rng):
        network = ActorCritic(activation=config["ACTIVATION"])
        rng, init_key = jax.random.split(rng)
        init_obs = jnp.zeros((1, 1))
        params = network.init(init_key, init_obs)
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(
                linear_schedule if config["ANNEAL_LR"] else config["LR"], eps=1e-5
            ),
        )
        state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

        rng, env_key = jax.random.split(rng)
        reset_keys = jax.random.split(env_key, config["NUM_ENVS"])
        env_states, obs = batch_reset(reset_keys, env_params)

        def _update_step(runner_state, unused):
            state, env_states, obs, rng = runner_state

            def _env_step(carry, unused):
                state, env_states, obs, rng = carry
                rng, act_key = jax.random.split(rng)
                pi, value = state.apply_fn(state.params, obs)
                action = pi.sample(seed=act_key)
                log_prob = pi.log_prob(action)
                a_real = jnp.where(action == 1, 1.0, -1.0)

                next_env_states, next_obs, reward, done = batch_step(
                    env_states, a_real, env_params
                )

                transition = Transition(done, action, value, reward, log_prob, obs)
                carry = (state, next_env_states, next_obs, rng)
                return carry, transition

            runner_state, traj = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            state, env_states, obs, rng = runner_state
            _, last_val = state.apply_fn(state.params, obs)

            def _calculate_gae(traj, last_val):
                def _scan_fn(carry, transition):
                    gae, next_value = carry
                    delta = (
                        transition.reward
                        + config["GAMMA"] * next_value * (1 - transition.done)
                        - transition.value
                    )
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - transition.done) * gae
                    return (gae, transition.value), gae

                _, advantages = jax.lax.scan(
                    _scan_fn,
                    (jnp.zeros_like(last_val), last_val),
                    traj,
                    reverse=True,
                    unroll=8,
                )
                return advantages, advantages + traj.value

            advantages, targets = _calculate_gae(traj, last_val)

            def _update_epoch(update_state, unused):
                def _update_minibatch(train_state, batch):
                    traj, advantages, targets = batch

                    def _loss_fn(params, traj, gae, targets):
                        pi, value = train_state.apply_fn(params, traj.obs)
                        log_prob = pi.log_prob(traj.action)
                        ratio = jnp.exp(log_prob - traj.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)

                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

                        value_pred_clipped = traj.value + (
                            value - traj.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        entropy = pi.entropy().mean()
                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss

                    loss, grads = jax.value_and_grad(_loss_fn)(
                        train_state.params, traj, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, loss

                train_state, traj, advantages, targets, rng = update_state
                rng, shuffle_key = jax.random.split(rng)
                batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
                permutation = jax.random.permutation(shuffle_key, batch_size)

                batch = (traj, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled,
                )

                train_state, _ = jax.lax.scan(_update_minibatch, train_state, minibatches)
                update_state = (train_state, traj, advantages, targets, rng)
                return update_state, None

            update_state = (state, traj, advantages, targets, rng)
            update_state, _ = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            state = update_state[0]
            return (state, env_states, obs, rng), traj.reward.mean()

        runner_state = (state, env_states, obs, rng)
        runner_state, rewards = jax.lax.scan(_update_step, runner_state, None, num_updates)
        final_state, _, _, _ = runner_state
        return rewards, final_state.params  # return params for evaluation

    return jax.jit(train)


def collect_returns(params, config, env_params, seed, max_env_steps):
    net = ActorCritic(activation=config["ACTIVATION"])
    rng = jax.random.PRNGKey(seed + 123)
    keys = jax.random.split(rng, config["NUM_ENVS"])
    env_states, obs = batch_reset(keys, env_params)
    ep_ret = jnp.zeros(config["NUM_ENVS"])
    ep_returns, ep_steps = [], []
    steps = 0
    while steps < max_env_steps:
        pi, _ = net.apply(params, obs)
        act = pi.mode()               # deterministic greedy action
        a_real = jnp.where(act == 1, 1.0, -1.0)
        next_states, next_obs, rew, done = batch_step(env_states, a_real, env_params)
        ep_ret = ep_ret + rew
        steps += config["NUM_ENVS"]
        if bool(jnp.any(done)):
            finished = np.asarray(ep_ret[np.where(np.asarray(done))])
            ep_returns.append(float(np.mean(finished)))
            ep_steps.append(steps)
            # reset finished envs
            reset_count = int(np.asarray(done).sum())
            rng, subk = jax.random.split(rng)
            new_keys = jax.random.split(subk, reset_count)
            new_states, new_obs = batch_reset(new_keys, env_params)
            s = np.array(next_states.s); t = np.array(next_states.t); ke = np.array(next_states.key)
            ob = np.array(next_obs); er = np.array(ep_ret)
            idx = np.where(np.asarray(done))[0]
            s[idx] = np.array(new_states.s); t[idx] = np.array(new_states.t)
            ke[idx] = np.array(new_states.key); ob[idx] = np.array(new_obs)
            er[idx] = 0.0
            env_states = env_states._replace(s=jnp.array(s), t=jnp.array(t), key=jnp.array(ke))
            obs = jnp.array(ob); ep_ret = jnp.array(er)
        else:
            env_states, obs = next_states, next_obs
    return np.column_stack([np.array(ep_steps, dtype=np.int64),
                            np.array(ep_returns, dtype=np.float32)])


if __name__ == "__main__":
    difficulty = os.getenv("CHAIN_DIFFICULTY", "medium")
    total_steps = int(os.getenv("CHAIN_TOTAL_ENV_STEPS", 80000))
    seed = int(os.getenv("CHAIN_SEED", 0))

    env_params = DIFFICULTIES[difficulty]
    config = {
        "TOTAL_ENV_STEPS": total_steps,
        "NUM_ENVS": 8,
        "NUM_STEPS": 64,
        "NUM_MINIBATCHES": 4,
        "UPDATE_EPOCHS": 4,
        "LR": 2.5e-4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": True,
    }

    print(f"Running PPO | Difficulty={difficulty} | Steps={total_steps}")
    rng = jax.random.PRNGKey(seed)
    train_fn = make_train(config, env_params)
    rewards, final_params = train_fn(rng)

    # Collect episodic returns vs env steps post-training
    eval_curves = collect_returns(final_params, config, env_params, seed, max_env_steps=total_steps)

    save_dir = f"runs/{ALGO_NAME}"
    os.makedirs(save_dir, exist_ok=True)
    path = f"{save_dir}/{difficulty}.csv"
    path_eval = f"{save_dir}/{difficulty}_eval.csv"
    np.savetxt(path_eval, eval_curves, fmt=["%d","%.6f"], delimiter=",")
    np.savetxt(path, eval_curves, fmt=["%d","%.6f"], delimiter=",")
    print(f"Saved episodic returns to {path}")
