# external/purejaxql_pqn/pqn_learner.py
# ==========================================================
# Parameterized Q-Network (PQN) Learner — JAX / Flax
# PureJaxQL-style: no target net, no fingerprints,
# RAdam + grad clip; per-minibatch SGD on provided λ-targets.
# ==========================================================
from typing import Sequence
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState

class QNetwork(nn.Module):
    hidden_dims: Sequence[int]
    n_actions: int
    @nn.compact
    def __call__(self, obs):
        x = obs
        for h in self.hidden_dims:
            x = nn.relu(nn.Dense(h)(x))
        return nn.Dense(self.n_actions)(x)

def _sgd_step(state: TrainState, obs, act, target):
    # obs: (B, obs_dim) | act: (B,) int | target: (B,)
    def loss_fn(params):
        q = state.apply_fn({"params": params}, obs)           # (B, A)
        # gather Q(s,a)
        q_sa = jnp.take_along_axis(q, act[:, None], axis=-1).squeeze(-1)
        loss = 0.5 * jnp.mean((q_sa - target) ** 2)
        return loss
    grads = jax.grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state

_sgd_step = jax.jit(_sgd_step)  # match PJQL speed

class PQNLearner:
    def __init__(
        self, seed, obs_dim, n_actions,
        hidden_dims=(128, 128),
        lr=2.5e-4, max_grad_norm=10.0
    ):
        self.n_actions = n_actions
        self.rng = jax.random.PRNGKey(seed)

        net = QNetwork(hidden_dims, n_actions)
        params = net.init(self.rng, jnp.zeros((1, obs_dim)))["params"]

        # PureJaxQL style: grad clip + RAdam
        tx = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.radam(learning_rate=lr),
        )
        self.q = TrainState.create(apply_fn=net.apply, params=params, tx=tx)

    def sample_action(self, obs, eps):
        # obs: (B, obs_dim)  | eps: scalar float
        self.rng, k1, k2 = jax.random.split(self.rng, 3)
        q = self.q.apply_fn({"params": self.q.params}, obs)   # (B, A)
        greedy = jnp.argmax(q, axis=-1)
        rand = jax.random.randint(k1, greedy.shape, 0, self.n_actions)
        explore = jax.random.uniform(k2, greedy.shape) < eps
        return jnp.where(explore, rand, greedy)

    def train_minibatch(self, obs, act, target):
        self.q = _sgd_step(self.q, obs, act, target)
