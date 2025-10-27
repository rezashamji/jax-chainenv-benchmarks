#external/jaxrl2_sac/sac_learner.py
"""
Canonical Soft Actor-Critic (SAC, fixed α).
- Twin Q critics with Polyak targets
- Stochastic tanh-Gaussian policy (reparameterization)
- Fixed temperature α (easy to extend to auto-tuning later)
"""

import functools
from typing import Sequence
import jax
import jax.numpy as jnp
import optax
from jax import tree_util


# ----------------------------
# Target update (Polyak)
# ----------------------------
def target_update(source_params, target_params, tau: float):
    return tree_util.tree_map(lambda s, t: tau * s + (1 - tau) * t, source_params, target_params)


# ----------------------------
# Policy: tanh-Gaussian
# ----------------------------
def sample_action_and_logp(params, obs, rng):
    """Reparameterized tanh-Normal with correct log-prob through tanh."""
    for w, b in params[:-2]:
        obs = jnp.tanh(obs @ w + b)
    w_mu, b_mu = params[-2]
    w_logstd, b_logstd = params[-1]

    mu = obs @ w_mu + b_mu
    log_std = jnp.clip(obs @ w_logstd + b_logstd, -5, 2)
    std = jnp.exp(log_std)

    eps = jax.random.normal(rng, mu.shape)
    pre_tanh = mu + eps * std
    a = jnp.tanh(pre_tanh)

    # change of variables correction
    logp = -0.5 * ((eps ** 2) + 2 * log_std + jnp.log(2 * jnp.pi))
    logp = logp.sum(-1) - jnp.log(jnp.clip(1 - a**2, 1e-6, 1)).sum(-1)
    return a, logp


# ----------------------------
# One SAC update step
# ----------------------------
@functools.partial(
    jax.jit,
    static_argnames=("actor_apply", "critic_apply", "actor_tx", "critic_tx"),
)
def _update_jit(
    actor_params,
    critic_params,
    target_critic_params,
    batch,
    discount,
    tau,
    actor_apply,
    critic_apply,
    actor_opt_state,
    critic_opt_state,
    actor_tx,
    critic_tx,
    alpha,
    rng1,  # fresh RNG for target action
    rng2,  # fresh RNG for actor loss
):
    # ---- target with CURRENT actor ----
    next_a, next_logp = actor_apply(actor_params, batch["next_observations"], rng1)
    next_q1, next_q2 = critic_apply(target_critic_params, batch["next_observations"], next_a)
    target_q = batch["rewards"] + discount * batch["masks"] * (
        jnp.minimum(next_q1, next_q2) - alpha * next_logp[:, None]
    )
    target_q = jax.lax.stop_gradient(target_q)  # clarity: no grad through target

    # ---- critic update ----
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

    new_target_critic_params = target_update(new_critic_params, target_critic_params, tau)

    # ---- actor update ----
    def actor_loss_fn(params):
        a, logp = actor_apply(params, batch["observations"], rng2)
        q1, q2 = critic_apply(new_critic_params, batch["observations"], a)
        q = jnp.minimum(q1, q2)
        loss = (alpha * logp[:, None] - q).mean()
        return loss, {"actor_loss": loss}

    (actor_loss, actor_info), actor_grads = jax.value_and_grad(
        actor_loss_fn, has_aux=True
    )(actor_params)
    actor_updates, new_actor_opt_state = actor_tx.update(
        actor_grads, actor_opt_state, actor_params
    )
    new_actor_params = optax.apply_updates(actor_params, actor_updates)

    info = {**critic_info, **actor_info}
    return (
        new_actor_params,
        new_critic_params,
        new_target_critic_params,
        info,
        new_actor_opt_state,
        new_critic_opt_state,
    )


class SACLearner:
    """SAC learner (fixed α)."""

    def __init__(
        self,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (64, 64),
        discount: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
    ):
        self.discount = discount
        self.tau = tau
        self.alpha = alpha
        self.step = 1

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key1, critic_key2 = jax.random.split(rng, 4)

        obs_dim = observations.shape[-1]
        act_dim = actions.shape[-1]

        # networks (pure JAX MLPs)
        def actor_apply(params, obs, rng):
            return sample_action_and_logp(params, obs, rng)

        def critic_apply(params, obs, act):
            def single_head(p, o, a):
                x = jnp.concatenate([o, a], -1)
                for w, b in p[:-1]:
                    x = jax.nn.relu(x @ w + b)
                w, b = p[-1]
                return x @ w + b
            q1 = single_head(params[0], obs, act)
            q2 = single_head(params[1], obs, act)
            return q1, q2

        def init_mlp(rng, in_dim, out_dim, hidden):
            layers = []
            dims = [in_dim] + list(hidden) + [out_dim]
            for i in range(len(dims) - 1):
                rng, subk = jax.random.split(rng)
                w = jax.random.normal(subk, (dims[i], dims[i + 1])) * 0.1
                b = jnp.zeros((dims[i + 1],))
                layers.append((w, b))
            return tuple(layers)

        def init_gaussian_actor(rng, in_dim, hidden, out_dim):
            layers = []
            dims = [in_dim] + list(hidden)
            for i in range(len(dims) - 1):
                rng, subk = jax.random.split(rng)
                w = jax.random.normal(subk, (dims[i], dims[i + 1])) * 0.1
                b = jnp.zeros((dims[i + 1],))
                layers.append((w, b))
            rng, k1, k2 = jax.random.split(rng, 3)
            w_mu = jax.random.normal(k1, (dims[-1], out_dim)) * 0.1
            b_mu = jnp.zeros((out_dim,))
            w_logstd = jax.random.normal(k2, (dims[-1], out_dim)) * 0.1
            b_logstd = jnp.zeros((out_dim,))
            layers += [(w_mu, b_mu), (w_logstd, b_logstd)]
            return tuple(layers)

        self.actor_params = init_gaussian_actor(actor_key, obs_dim, hidden_dims, act_dim)
        self.critic_params = (
            init_mlp(critic_key1, obs_dim + act_dim, 1, hidden_dims),
            init_mlp(critic_key2, obs_dim + act_dim, 1, hidden_dims),
        )
        self.target_critic_params = self.critic_params

        self.actor_tx = optax.adam(actor_lr)
        self.critic_tx = optax.adam(critic_lr)
        self.actor_opt_state = self.actor_tx.init(self.actor_params)
        self.critic_opt_state = self.critic_tx.init(self.critic_params)

        self.actor_apply = actor_apply
        self.critic_apply = critic_apply
        self.rng = rng

    # -------- batch-safe action sampling --------
    def sample_actions(self, obs: jnp.ndarray) -> jnp.ndarray:
        """
        Accepts (obs_dim,) or (B, obs_dim). Uses per-sample RNG.
        Returns actions with matching leading shape.
        """
        obs = jnp.atleast_2d(obs)   # (B, obs_dim)
        B = obs.shape[0]
        self.rng, subkey = jax.random.split(self.rng)
        keys = jax.random.split(subkey, B)

        def _one(o, k):
            a, _ = self.actor_apply(self.actor_params, o, k)  # (act_dim,)
            return a

        actions = jax.vmap(_one)(obs, keys)  # (B, act_dim)
        return actions if B > 1 else actions[0]

    def eval_actions(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Deterministic SAC action: tanh(mu) without sampling."""
        obs = jnp.atleast_2d(obs)
        x = obs
        for w, b in self.actor_params[:-2]:
            x = jnp.tanh(x @ w + b)
        w_mu, b_mu = self.actor_params[-2]
        mu = x @ w_mu + b_mu
        a = jnp.tanh(mu)
        return a

    
    def update(self, batch):
        self.step += 1
        self.rng, rng1, rng2 = jax.random.split(self.rng, 3)

        (
            self.actor_params,
            self.critic_params,
            self.target_critic_params,
            info,
            self.actor_opt_state,
            self.critic_opt_state,
        ) = _update_jit(
            self.actor_params,
            self.critic_params,
            self.target_critic_params,
            batch,
            self.discount,
            self.tau,
            self.actor_apply,
            self.critic_apply,
            self.actor_opt_state,
            self.critic_opt_state,
            self.actor_tx,
            self.critic_tx,
            self.alpha,
            rng1,
            rng2,
        )
        return info
