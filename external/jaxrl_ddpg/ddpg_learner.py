#external/jaxrl_ddpg/ddpg_learner.py
"""
Self-contained JAX DDPG learner (faithful to the classic algorithm).
- Actor: deterministic μ(s) with tanh squash → action in [-1, 1].
- Critic: twin Q-networks (Q1, Q2) for stability; policy uses Q1 for the objective.
- Targets: Polyak-averaged critic *and* target actor for bootstrap.
- JIT: one fused `_update_jit` step for speed.
"""

import functools
from typing import Sequence
import jax
import jax.numpy as jnp
import optax
from jax import tree_util

# ----------------------------
# Polyak averaging
# ----------------------------
def target_update(source_params, target_params, tau: float):
    """Exponential moving average for target networks.

    new_target = tau * source + (1 - tau) * old_target
    Small `tau` (e.g., 0.005) → slow, stable target drift.
    """

    return tree_util.tree_map(lambda s, t: tau * s + (1 - tau) * t, source_params, target_params)

# ----------------------------
# One DDPG update (critic → actor → targets)
# ----------------------------
@functools.partial(
    jax.jit,
    static_argnames=("actor_apply", "critic_apply", "update_target", "actor_tx", "critic_tx"),
)
def _update_jit(
    actor_params,
    actor_target_params,
    critic_params,
    target_params,
    batch,
    discount,
    tau,
    update_target,
    actor_apply,
    critic_apply,
    actor_opt_state,
    critic_opt_state,
    actor_tx,
    critic_tx,
):
    """Single off-policy gradient update.


    1) Critic target: r + γ * Q_target1(s', μ_target(s'))
    2) Critic loss: MSE((Q1, Q2) vs target)
    3) Actor loss: maximize Q1(s, μ(s)) → minimize -Q1
    4) Polyak update target networks
    """
    # ---- Critic target with TARGET actor/critic ----
    next_actions = actor_apply(actor_target_params, batch["next_observations"])
    next_q1, _ = critic_apply(target_params, batch["next_observations"], next_actions)
    target_q = batch["rewards"] + discount * batch["masks"] * next_q1
    
    # ---- Critic update ----   
    def critic_loss_fn(params):
        q1, q2 = critic_apply(params, batch["observations"], batch["actions"])
        loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
        return loss, {"critic_loss": loss}

    (critic_loss, critic_info), critic_grads = jax.value_and_grad(
        critic_loss_fn, has_aux=True
    )(critic_params)
    critic_updates, new_critic_opt_state = critic_tx.update(
        critic_grads, critic_opt_state, critic_params
    )
    new_critic_params = optax.apply_updates(critic_params, critic_updates)

    # Optionally refresh critic target (every `target_update_period` steps)
    new_target_params = jax.lax.cond(
        update_target,
        lambda _: target_update(new_critic_params, target_params, tau),
        lambda _: target_params,
        operand=None,
    )

    # ---- Actor update (maximize Q1 via deterministic policy gradient) ----
    def actor_loss_fn(params):
        actions = actor_apply(params, batch["observations"])
        q1, _ = critic_apply(new_critic_params, batch["observations"], actions)
        loss = -q1.mean()
        return loss, {"actor_loss": loss}

    (actor_loss, actor_info), actor_grads = jax.value_and_grad(
        actor_loss_fn, has_aux=True
    )(actor_params)
    actor_updates, new_actor_opt_state = actor_tx.update(
        actor_grads, actor_opt_state, actor_params
    )
    new_actor_params = optax.apply_updates(actor_params, actor_updates)
    
    # Target actor update (Polyak)
    new_actor_target_params = target_update(new_actor_params, actor_target_params, tau)

    info = {**critic_info, **actor_info}
    return (
        new_actor_params,
        new_actor_target_params,
        new_critic_params,
        new_target_params,
        info,
        new_actor_opt_state,
        new_critic_opt_state,
    )
# ----------------------------
# Learner wrapper (init + APIs)
# ----------------------------
class DDPGLearner:
    """Lightweight DDPG learner with pure-JAX MLPs.


    Exposes:
    - `sample_actions(obs)` → exploration actions with Gaussian noise
    - `actor_apply(params, obs)` → deterministic μ(s) for evaluation
    - `update(batch)` → one gradient step using off-policy batch
    """

    def __init__(
        self,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        hidden_dims: Sequence[int] = (64, 64),
        discount: float = 0.99,
        tau: float = 0.005,
        target_update_period: int = 1,
        exploration_noise: float = 0.02,
    ):
        self.discount = discount
        self.tau = tau
        self.target_update_period = target_update_period
        self.exploration_noise = exploration_noise
        self.step = 1

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key1, critic_key2 = jax.random.split(rng, 4)

        obs_dim = observations.shape[-1]
        act_dim = actions.shape[-1]

        # ----- Network defs (pure JAX MLPs) -----
        def actor_apply(params, obs):
            # tanh nonlinearity; final tanh squashes to [-1, 1]
            for w, b in params[:-1]:
                obs = jnp.tanh(obs @ w + b)
            w, b = params[-1]
            return jnp.tanh(obs @ w + b)

        def critic_apply(params, obs, act):
            """Twin critics; each sees [obs, act]."""
            def single_head(p, o, a):
                x = jnp.concatenate([o, a], -1)
                for w, b in p[:-1]:
                    x = jax.nn.relu(x @ w + b)
                w, b = p[-1]
                return x @ w + b

            q1 = single_head(params[0], obs, act)
            q2 = single_head(params[1], obs, act)
            return q1, q2

        # ----- Parameter init -----
        def init_mlp(rng, in_dim, out_dim, hidden):
            layers = []
            dims = [in_dim] + list(hidden) + [out_dim]
            for i in range(len(dims) - 1):
                rng, subk = jax.random.split(rng)
                w = jax.random.normal(subk, (dims[i], dims[i + 1])) * 0.1
                b = jnp.zeros((dims[i + 1],))
                layers.append((w, b))
            return tuple(layers)

        # Actor and twin critics + their targets
        self.actor_params = init_mlp(actor_key, obs_dim, act_dim, hidden_dims)
        self.actor_target_params = tree_util.tree_map(lambda x: x, self.actor_params)
        self.critic_params = (
            init_mlp(critic_key1, obs_dim + act_dim, 1, hidden_dims),
            init_mlp(critic_key2, obs_dim + act_dim, 1, hidden_dims),
        )
        self.target_params = self.critic_params

        # Optimizers
        self.actor_tx = optax.adam(actor_lr)
        self.critic_tx = optax.adam(critic_lr)
        self.actor_opt_state = self.actor_tx.init(self.actor_params)
        self.critic_opt_state = self.critic_tx.init(self.critic_params)

        # API handles
        self.actor_apply = actor_apply
        self.critic_apply = critic_apply
        self.rng = rng

    # ----- Action sampling with exploration noise -----
    def sample_actions(self, obs: jnp.ndarray, temperature: float = 1.0) -> jnp.ndarray:
        """Return actions for a batch of observations with Gaussian noise.
        Noise scale = `exploration_noise * temperature`.
        """
        obs = jnp.atleast_2d(obs)
        B = obs.shape[0]
        self.rng, subkey = jax.random.split(self.rng)
        keys = jax.random.split(subkey, B)

        def _one(o, k):
            a = self.actor_apply(self.actor_params, o)
            n = jax.random.normal(k, a.shape) * self.exploration_noise * temperature
            return jnp.clip(a + n, -1.0, 1.0)

        return jax.vmap(_one)(obs, keys)


    # ----- One learner update -----
    def update(self, batch):
        self.step += 1
        (
            self.actor_params,
            self.actor_target_params,
            self.critic_params,
            self.target_params,
            info,
            self.actor_opt_state,
            self.critic_opt_state,
        ) = _update_jit(
            self.actor_params,
            self.actor_target_params,
            self.critic_params,
            self.target_params,
            batch,
            self.discount,
            self.tau,
            self.step % self.target_update_period == 0,
            self.actor_apply,
            self.critic_apply,
            self.actor_opt_state,
            self.critic_opt_state,
            self.actor_tx,
            self.critic_tx,
        )
        return info
