from typing import Protocol, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import jax, jax.numpy as jnp
import optax

class ScoreModel(Protocol):
    def __call__(self, t: jax.Array, x: jax.Array) -> jax.Array:
        """Outputs the predicted noise epsilon or the score."""
        ...

@dataclass(frozen=True)
class ScoreMatching:
    num_timesteps: int
    sigma_t_val: float = 0.1 # Constant or time-dependent noise scale sigma_t
    sigma_min: float = 1e-4
    action_dim: int = 1
    include_entropy: bool = True

    def get_xt(self, x_start: jax.Array, x_noise: jax.Array, t: jax.Array) -> jax.Array:
        """
        Optimal Transport Path (Linear):
        x_t = t * x_start + (1 - t) * x_noise
        To avoid singularity at t=0, we often use a small sigma_min offset.
        """
        t_obj = t.reshape(-1, *([1] * (len(x_start.shape) - 1)))
        # Standard OT interpolation
        x_t = t_obj * x_start + (1.0 - t_obj) * x_noise
        return x_t
    
    def get_ut(self, x_start: jax.Array, x_noise: jax.Array) -> jax.Array:
        """Target Velocity for CFM: u_t = x_start - x_noise"""
        return x_start - x_noise

    def solve_sde(self, model: ScoreModel, shape: Tuple[int, ...], key: jax.Array, sigma_t: float=None) -> jax.Array:
        """
        Sample action by solving the SDE using Euler-Maruyama 
        Update: x_{t+dt} = x_t + [v_theta + (sigma_t^2 / 2t) * (x_t + (1-t)v_theta)] * dt + sigma_t * sqrt(dt) * eps
        """
        s_t = sigma_t if sigma_t is not None else self.sigma_t_val

        k1, k2 = jax.random.split(key)
        x = jax.random.normal(k1, shape)
        dt = 1.0 / self.num_timesteps

        def body_fn(carry, t_val):
            x, key_step = carry
            t = jnp.array(t_val)
            k_step, k_next = jax.random.split(key_step)
            
            # Predict Velocity Field v_theta(x, t)
            v_theta = model(t, x)
            
            # 1. Calculate Drift Term based on Eq. 28/29
            # drift = v_theta + (sigma_t^2 / 2t) * (x + (1-t)v_theta)
            sigma_sq = s_t ** 2
            correction_term = (sigma_sq / (2.0 * jnp.maximum(t, 1e-4))) * (x + (1.0 - t) * v_theta)
            drift = v_theta + correction_term
            
            # 2. Diffusion / Stochasticity
            # sigma_t * sqrt(dt) * epsilon
            eps = jax.random.normal(k_step, shape)
            diffusion = s_t * jnp.sqrt(dt) * eps
            
            # 3. Apply Update Rule (Eq. 29)
            x_next = x + drift * dt + diffusion
            
            return (x_next, k_next), None

        # Solve from t=eps (noise) to t=1 (data)
        timesteps = jnp.linspace(self.sigma_min, 1.0, self.num_timesteps)
        (x_final, _), _ = jax.lax.scan(body_fn, (x, k2), timesteps)
        
        return jnp.clip(x_final, -1.0, 1.0)

    def compute_cfm_loss(self, model: ScoreModel, t: jax.Array, x_t: jax.Array, x_start: jax.Array, x_noise: jax.Array, q_weights: jax.Array) -> jax.Array:
        """CFM Loss: Regress velocity field directly as per Flow Matching"""
        target_v = self.get_ut(x_start, x_noise)
        pred_v = model(t, x_t)
        loss = jnp.mean(jnp.square(pred_v - target_v) * q_weights[:, None])
        return loss

    def compute_loss(self, key: jax.Array, model: ScoreModel, q_weights: jax.Array, x_start: jax.Array, log_alpha: jax.Array) -> jax.Array:
        x_start = jax.lax.stop_gradient(x_start)
        q_weights = jax.lax.stop_gradient(q_weights)
        log_alpha = jax.lax.stop_gradient(log_alpha)

        # Standard loss setup remains same, but uses the OT path logic
        batch_size = x_start.shape[0]
        k1, k2, k3 = jax.random.split(key, 3)
        
        t = jax.random.uniform(k1, (batch_size,), minval=self.sigma_min, maxval=1.0)
        x_noise = jax.random.normal(k2, x_start.shape)
        x_t = self.get_xt(x_start, x_noise, t)
        
        cfm_loss = self.compute_cfm_loss(model, t, x_t, x_start, x_noise, q_weights)

        if not self.include_entropy:
            return cfm_loss, None
           
        # Divergence-based entropy loss (using the SDE drift field)
        entropy_loss = self.compute_proxy_entropy_loss(k3, model, x_t, t)
        total_policy_loss = cfm_loss + jnp.exp(log_alpha) * entropy_loss
        return total_policy_loss, -entropy_loss

    def compute_proxy_entropy_loss(self, key: jax.Array, model: ScoreModel, x_t: jax.Array, t: jax.Array) -> jax.Array:
        """Hutchinson estimate for the SDE drift divergence"""
        def drift_fn(x):
            v_theta = model(t, x)
            t_obj = t.reshape(-1, 1)
            sigma_sq = self.sigma_t_val ** 2
            return v_theta + (sigma_sq / (2.0 * jnp.maximum(t_obj, 1e-4))) * (x + (1.0 - t_obj) * v_theta)

        x_t = jax.lax.stop_gradient(x_t)
        eps = jax.random.normal(key, x_t.shape)
        _, jvp_val = jax.jvp(drift_fn, (x_t,), (eps,))
        return -jnp.mean(jnp.sum(jvp_val * eps, axis=-1))