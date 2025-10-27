# envs/chain_jax_env.py
# ==========================================================
# Pure-JAX Chain environment with tunable exploration difficulty.
#   * Functional: no side effects; all state is explicit.
#   * JIT/vmap/lax.scan friendly (no Python control flow that depends on data).
#   * Observation is a 1D float array with the current position index.
# ==========================================================


from typing import NamedTuple, Dict
import jax
import jax.numpy as jnp


# ----------------------------------------------------------
# Core dataclasses (JAX-friendly, pytree-compatible)
# ----------------------------------------------------------

class EnvParams(NamedTuple):
    # Fixed environment configuration passed to reset/step.
    N: int           # chain length (positions are 0..N-1)
    H: int           # episode horizon (max steps before termination)
    slip: float      # probability of flipping the intended move direction
    r_small: float   # reward when visiting position 1 (local, tempting)
    r_big: float     # reward when reaching the last position (N-1, goal)

class EnvState(NamedTuple):
    # The evolving simulator state carried through step().
    s: jnp.int32     # current position in chain
    t: jnp.int32     # elapsed step counter
    key: jax.Array   # PRNG key for stochastic transitions

# ----------------------------------------------------------
# Environment dynamics
# ----------------------------------------------------------
def reset(key: jax.Array, params: EnvParams):
    """Start a new episode.

    Args:
        key: PRNGKey for stochasticity (each env gets its own).
        params: static configuration.

    Returns:
        state: EnvState with position=0, time=0, fresh key.
        obs: Observation array shape (1,) with the current position.
    """
        
    state = EnvState(s=jnp.int32(0), t=jnp.int32(0), key=key)
    obs = jnp.array([0.0], dtype=jnp.float32)  # shape (1,)
    return state, obs


def step(state: EnvState, action: jnp.ndarray, params: EnvParams):
    """Advance one step.

    Action semantics:
      - We treat action > 0 as "move right", otherwise "move left".
      - With probability `slip`, the move direction is flipped (exploration pain).

    Termination:
      - Episode ends if we reach position N-1 (goal) OR when t hits H.

    Rewards:
      - Small reward at position 1 (local optimum).
      - Big reward at the last position (N-1).

    All operations are JAX ops, so this is JIT/vmap safe.
    """
        
    # Split the key: keep `key` to carry forward; use `subk` for this step's randomness.
    key, subk = jax.random.split(state.key)
    
    # Map real-valued action to a direction {-1, +1}.
    move = jnp.where(action > 0, 1, -1)

    # With probability `slip`, flip the direction.
    slip = jax.random.bernoulli(subk, params.slip)
    move = jnp.where(slip, -move, move)
    
    # Update position, clamped to [0, N-1].
    s_next = jnp.clip(state.s + move, 0, params.N - 1)

    # Compute reward: local reward at position 1, big reward at goal.
    reward = (
        jnp.where(s_next == 1, params.r_small, 0.0)
        + jnp.where(s_next == params.N - 1, params.r_big, 0.0)
    )
    
    # Time update and termination condition.
    t_next = state.t + 1
    done = (t_next >= params.H) | (s_next == params.N - 1)

    # Package next state and observation.
    next_state = EnvState(s=jnp.int32(s_next), t=t_next, key=key)
    obs = jnp.array([s_next], dtype=jnp.float32)  # shape (1,)
    return next_state, obs, reward, done

# ----------------------------------------------------------
# Difficulty presets (used by all algorithms)
# ----------------------------------------------------------
DIFFICULTIES: Dict[str, EnvParams] = {
    # Easier: shorter chain, low slip, noticeable local reward.
    "easy":   EnvParams(N=5,  H=15, slip=0.00, r_small=0.3, r_big=1.0),
    # Medium: longer chain, more slip, smaller local reward.
    "medium": EnvParams(N=7,  H=20, slip=0.15, r_small=0.1, r_big=1.0),
    # Hard: longest chain, high slip, no local lure to help exploration.
    "hard":   EnvParams(N=9,  H=25, slip=0.25, r_small=0.0, r_big=1.0),
}


# ----------------------------------------------------------
# Vectorized wrappers (batched reset/step across many envs)
# ----------------------------------------------------------
def batch_reset(keys, params: EnvParams):
    """Reset a batch of envs: keys shape (B, 2), returns (states_B, obs_B)."""
    return jax.vmap(reset, in_axes=(0, None))(keys, params)

def batch_step(states, actions, params: EnvParams):
    """Step a batch of envs: states_B & actions_B -> next_states_B, obs_B, rew_B, done_B."""
    return jax.vmap(step, in_axes=(0, 0, None))(states, actions, params)
