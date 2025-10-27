# algorithms/jax_buffer.py
# ==========================================================
# Minimal ring buffers implemented with JAX arrays.
#
# Intent:
#   * Keep the data on device as JAX arrays (good for slicing / vmapping).
#   * Simple "host-side" mutability (assigning .at[idx].set(...))—we don't JIT
#     `add_batch`/`sample` themselves. They’re meant to be used outside jit/scans.
#   * Returned batches are JAX arrays ready to feed into learners.
#
# Notes:
#   * `mask` follows the common convention mask = 1 - done (1 if nonterminal).
#   * This buffer is generic; PPO/PQN are on-policy and may not use it directly,
#     while DDPG/SAC can use it for off-policy training.
# ==========================================================

import jax
import jax.numpy as jnp
from typing import Dict, Tuple

class JaxReplayBuffer:
    """Ring buffer for continuous-control algorithms (e.g., SAC/DDPG).

    Layout (capacity, ...):
      obs      : (C, obs_dim)
      act      : (C, act_dim)
      rew      : (C, 1)
      mask     : (C, 1)     # mask = 1.0 if not done, else 0.0
      next_obs : (C, obs_dim)

    This class stores JAX arrays and updates them with .at[...] to keep things
    device-friendly. It tracks `pos` (write cursor) and logical `size`.
    """
        
    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = int(capacity)
        self.obs      = jnp.zeros((capacity, obs_dim), dtype=jnp.float32)
        self.act      = jnp.zeros((capacity, act_dim), dtype=jnp.float32)
        self.rew      = jnp.zeros((capacity, 1),       dtype=jnp.float32)
        self.mask     = jnp.zeros((capacity, 1),       dtype=jnp.float32)
        self.next_obs = jnp.zeros((capacity, obs_dim), dtype=jnp.float32)
        self.size     = 0
        self.pos      = 0

    def add_batch(self, obs, act, rew, mask, next_obs):
        """Append a batch of transitions at the current cursor (ring behavior).

        Args (all JAX arrays):
            obs      : (B, obs_dim)
            act      : (B, act_dim)
            rew      : (B,) or (B, 1)
            mask     : (B,) or (B, 1)   # 1.0 for nonterminal, 0.0 if done
            next_obs : (B, obs_dim)
        """
                
        B = obs.shape[0]
        # Compute the target indices for the slice [pos, pos+B) modulo capacity.
        idx = (jnp.arange(B) + self.pos) % self.capacity

        # In-place (JAX) writes for each field.
        self.obs      = self.obs.at[idx].set(obs)
        self.act      = self.act.at[idx].set(act)
        self.rew      = self.rew.at[idx].set(rew.reshape(B, 1))
        self.mask     = self.mask.at[idx].set(mask.reshape(B, 1))
        self.next_obs = self.next_obs.at[idx].set(next_obs)
        
        # Advance cursor and grow logical size up to capacity.
        self.pos  = int((self.pos + B) % self.capacity)
        self.size = int(min(self.capacity, self.size + B))

    def sample(self, key: jax.Array, batch_size: int) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        """Uniformly sample a minibatch without replacement.

        Args:
            key        : PRNGKey for sampling indices.
            batch_size : number of transitions to return.

        Returns:
            idx   : sampled indices (useful for prioritized variants, etc.).
            batch : dict of JAX arrays with shapes:
                    - "obs"      : (B, obs_dim)
                    - "act"      : (B, act_dim)
                    - "rew"      : (B,)
                    - "mask"     : (B,)
                    - "next_obs" : (B, obs_dim)
        """
                
        size = max(1, self.size)
        idx = jax.random.permutation(key, size)[:batch_size]
        batch = {
            "obs":      self.obs[idx],
            "act":      self.act[idx],
            "rew":      self.rew[idx].squeeze(-1),
            "mask":     self.mask[idx].squeeze(-1),
            "next_obs": self.next_obs[idx],
        }
        return idx, batch


class JaxReplayBufferPQN:
    """Ring buffer variant for PQN-style data with fingerprints.

    Additional fields:
      fp  : (C, 2)   # e.g., [epsilon, t_frac] at time of collection
      nfp : (C, 2)   # next state's fingerprint (optional features)

    Note: PQN in your repo trains mostly on on-policy rollouts; this buffer is
    here if you want to cache & replay those sequences offline.
    """
    
    def __init__(self, capacity: int, obs_dim: int, n_actions: int):
        self.capacity = int(capacity)
        self.obs  = jnp.zeros((capacity, obs_dim), dtype=jnp.float32)
        self.fp   = jnp.zeros((capacity, 2),       dtype=jnp.float32)
        self.act  = jnp.zeros((capacity, 1),       dtype=jnp.int32)
        self.rew  = jnp.zeros((capacity, 1),       dtype=jnp.float32)
        self.mask = jnp.zeros((capacity, 1),       dtype=jnp.float32)
        self.nobs = jnp.zeros((capacity, obs_dim), dtype=jnp.float32)
        self.nfp  = jnp.zeros((capacity, 2),       dtype=jnp.float32)
        self.size = 0
        self.pos  = 0

    def add_batch(self, obs, fp, act, rew, mask, nobs, nfp):
        """Append a batch; shapes follow field names (B, ...)."""
        B = obs.shape[0]
        idx = (jnp.arange(B) + self.pos) % self.capacity
        self.obs  = self.obs.at[idx].set(obs)
        self.fp   = self.fp.at[idx].set(fp)
        self.act  = self.act.at[idx].set(act.reshape(B, 1))
        self.rew  = self.rew.at[idx].set(rew.reshape(B, 1))
        self.mask = self.mask.at[idx].set(mask.reshape(B, 1))
        self.nobs = self.nobs.at[idx].set(nobs)
        self.nfp  = self.nfp.at[idx].set(nfp)
        self.pos  = int((self.pos + B) % self.capacity)
        self.size = int(min(self.capacity, self.size + B))

    def sample(self, key: jax.Array, batch_size: int):
        """Uniform sample; returns indices and a dict batch for training."""
        size = max(1, self.size)
        idx = jax.random.permutation(key, size)[:batch_size]
        batch = {
            "obs": self.obs[idx],
            "eps_tfrac": self.fp[idx],
            "act": self.act[idx],
            "rew": self.rew[idx],
            "mask": self.mask[idx],
            "next_obs": self.nobs[idx],
            "next_eps_tfrac": self.nfp[idx],
        }
        return idx, batch
