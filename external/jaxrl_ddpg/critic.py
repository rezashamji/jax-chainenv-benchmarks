#external/jaxrl_ddpg/critic.py
import jax
import jax.numpy as jnp
# Critic step: Build a stable TD target with target networks; fit both Q heads to that target with MSE.

@jax.jit
def update(actor, critic, target_critic, batch, discount: float):
    """
    One critic update step for DDPG using a target network.

    Args:
        actor:          Actor TrainState; used here *only* to compute next actions.
        critic:         Critic TrainState to be updated (twin heads).
        target_critic:  Polyak-averaged target of the critic (twin heads).
        batch:          Dict with JAX arrays:
                        - "observations":      (B, obs_dim)
                        - "actions":           (B, act_dim)
                        - "rewards":           (B, 1)
                        - "next_observations": (B, obs_dim)
                        - "masks":             (B, 1) where mask = 1 - done
        discount:       γ in [0, 1].

    Returns:
        new_critic: Updated critic TrainState.
        info:       Dict with logging metrics (loss, mean Qs).
    """

    # ---- Build TD target with target networks (no grad into targets) ----
    # Deterministic next action from the *current* actor parameters.
    next_actions = actor.apply_fn(actor.params, batch["next_observations"])
    
    # Evaluate target critics at (s', a') and take the minimum head (Double-Q).
    next_q1, next_q2 = target_critic.apply_fn(
        target_critic.params, batch["next_observations"], next_actions
    )
    next_q = jnp.minimum(next_q1, next_q2) # (B, 1)
    
    # TD target: r + γ * mask * next_q.  `mask = 0` for terminal transitions.
    # Gradients are taken w.r.t. critic_params below, so there is no path back
    # through `target_critic.params` or `actor.params`; adding stop_gradient is optional.
    target_q = batch["rewards"] + discount * batch["masks"] * next_q

    def critic_loss_fn(critic_params):
        # Current Q estimates for the replayed (s, a).
        q1, q2 = critic.apply_fn(critic_params, batch["observations"], batch["actions"])
        # MSE against the target for both heads; average them.
        critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
        return critic_loss, {
            "critic_loss": critic_loss,
            "q1": q1.mean(),
            "q2": q2.mean(),
        }
    # Compute gradients w.r.t. critic params and apply them.
    new_critic, info = critic.apply_gradient(critic_loss_fn)
    return new_critic, info
