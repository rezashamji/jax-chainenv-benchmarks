#external/jaxrl_ddpg/actor.py
import jax
import jax.numpy as jnp
#Actor step: Compute π(s); evaluate Q(s, π(s)); push π to maximize the (clipped) Q estimate.

@jax.jit
def update(actor, critic, batch):
    """
    One actor (policy) update step for DDPG.

    Args:
        actor:  A Flax TrainState-like object with:
                - params: current actor parameters
                - apply_fn(params, obs) -> actions in [-1, 1]
                - apply_gradient(grad_fn) -> (new_actor_state, info_dict)
        critic: A TrainState-like twin-Q critic used only for evaluation here.
                We read critic.params but do NOT update it in this function.
        batch:  Dict with JAX arrays:
                - "observations": shape (B, obs_dim)

    Returns:
        new_actor: updated actor TrainState
        info:      dict with scalar logging (e.g., {"actor_loss": ...})
    """
    def actor_loss_fn(actor_params):
        # 1) Get current policy's actions for the batch of observations.
        actions = actor.apply_fn(actor_params, batch["observations"])

        # 2) Evaluate both critics at (s, a). We DON'T backprop into critic here;
        #    we just read its value estimate to provide a learning signal.
        q1, q2 = critic.apply_fn(critic.params, batch["observations"], actions)
        
        # 3) Clipped Double-Q: use the smaller estimate to reduce positive bias.
        q = jnp.minimum(q1, q2)

        # 4) DDPG objective: maximize Q(s, π(s)). Optimizers minimize, so negate.
        actor_loss = -q.mean()

        # Returning a tuple (loss, metrics) lets TrainState log without extra passes.
        return actor_loss, {"actor_loss": actor_loss}

    # Compute gradients w.r.t. actor params and apply them in one go.
    new_actor, info = actor.apply_gradient(actor_loss_fn)
    return new_actor, info
