from typing import Protocol, Tuple, Optional, Literal
from dataclasses import dataclass

import numpy as np
import jax, jax.numpy as jnp
import optax
from functools import partial

class FlowModel(Protocol):
    def __call__(self, t: jax.Array, x: jax.Array, obs: jax.Array) -> jax.Array:
        ...

@dataclass(frozen=True)
class FlowMatching:
    num_timesteps: int
    action_dim: int = 1
    include_entropy: bool = True
    use_accurate_entropy: bool = False
    t_distribution: Literal["uniform", "logit_normal"] = "uniform"

    def get_xt(self, sigma_min: jax.Array, x_start: jax.Array, x_noise: jax.Array, t: jax.Array) -> jax.Array:
        """
        FM Path defined in the image:
        mu_t(x) = t * x_1
        sigma_t(x) = 1 - (1 - sigma_min) * t
        x_t = mu_t + sigma_t * x_0
        """
        # Reshape t for broadcasting
        t_obj = t.reshape(-1, *([1] * (len(x_start.shape) - 1)))
        
        # Calculate mu_t and sigma_t based on the formulas
        mu_t = t_obj * x_start
        sigma_t = 1.0 - (1.0 - sigma_min) * t_obj
        
        # Combine to get x_t
        x_t = mu_t + sigma_t * x_noise
        return x_t

    def get_ut(self, sigma_min: jax.Array, x_start: jax.Array, x_noise: jax.Array) -> jax.Array:
        """ 
        calculate Target Velocity:
        u_t = d/dt x_t = x_start + (sigma_min - 1) * x_noise
        """
        return x_start + (sigma_min - 1.0) * x_noise

    def solve_ode(self, model: FlowModel, shape: Tuple[int, ...], key: jax.Array, obs: jax.Array) -> jax.Array:
        """
        sample action by solving ODE
        """
        x = jax.random.normal(key, shape)
        dt = 1.0 / self.num_timesteps

        def body_fn(x, t_val):
            t = jnp.array(t_val)
            v = model(t, x, obs)
            x_next = x + v * dt
            return x_next, None

        timesteps = jnp.linspace(0.0, 1.0, self.num_timesteps) #Timesteps shape: (20,)
        x_final, _ = jax.lax.scan(body_fn, x, timesteps)
        
        return jnp.tanh(x_final)
        # return jnp.clip(x_final, -1.0, 1.0)

    def compute_cfm_loss(self, sigma_min: jax.Array, model: FlowModel, t: jax.Array, x_t: jax.Array, x_start: jax.Array, x_noise: jax.Array, q_weights: jax.Array, obs: jax.Array) -> jax.Array:
        """
        Corresponding to the Torch version of cfm_losses
        """
        # Get interpolation point x_t using new q_sample
        target_v = self.get_ut(sigma_min, x_start, x_noise)
        
        # Model prediction
        pred_v = model(t, x_t, obs)
    
        loss = jnp.mean(jnp.square(pred_v - target_v)*q_weights[:, None])
        return loss

    @staticmethod
    def _sample_logit_normal(
        key,
        batch_size,
        loc=0.0,
        scale=1.0,
    ) -> jax.Array:
        "sample t according to logit normal distribution"
        y = jax.random.normal(key, (batch_size,))
        y = loc + scale * y
        t = jax.nn.sigmoid(y)

        return t

    def compute_loss(self, sigma_min: jax.Array, key: jax.Array, model: FlowModel, q_weights: jax.Array, x_start: jax.Array,  log_alpha: jax.Array, obs: jax.Array) -> jax.Array:
        x_start = jax.lax.stop_gradient(x_start)
        q_weights = jax.lax.stop_gradient(q_weights)
        log_alpha = jax.lax.stop_gradient(log_alpha)

        batch_size = x_start.shape[0]
        k1, k2, k3, k4 = jax.random.split(key, 4)

        u_start = jnp.arctanh(jnp.clip(x_start, -0.999999, 0.999999))  # Inverse tanh to get pre-squashed actions
        
        # Sample time t ~ U(0, 1)
        if self.t_distribution == "uniform":
            t = jax.random.uniform(k1, (batch_size,))
        elif self.t_distribution == "logit_normal":
            t = self._sample_logit_normal(k1, batch_size)
        else:
            raise ValueError(f"Unknown t_distribution: {self.t_distribution}")
        u_noise = jax.random.normal(k2, u_start.shape)
        u_t = self.get_xt(sigma_min, u_start, u_noise, t)
        cfm_loss = self.compute_cfm_loss(sigma_min, model, t, u_t, u_start, u_noise, q_weights, obs)

        if not self.include_entropy:
            return cfm_loss, None
        else:
            entropy_loss, divergence = self.compute_proxy_entropy_loss(k3, model, u_noise, u_t, t, obs)
            total_policy_loss = cfm_loss +jnp.exp(log_alpha)*entropy_loss
            return total_policy_loss, cfm_loss, divergence

    def _get_single_divergence(self, model: FlowModel, u: jax.Array, t: jax.Array, obs: jax.Array, key: jax.Array) -> jax.Array:
        """
        Input: x shape (D,), t shape ()
        Output: scalar ()
        """
        def single_sample_fn(u_in):
            # model expects t: (B,), x: (B, D)
            return model(t[None], u_in[None, :], obs[None, :])[0]
        
        epsilon = jax.random.rademacher(key, shape=u.shape, dtype=u.dtype)
        v_u, jvp_eps = jax.jvp(single_sample_fn, (u,), (epsilon,))
        div_u = jnp.sum(epsilon * jvp_eps)

        # v_u = single_sample_fn(u)
        # div_u = jnp.trace(jax.jacrev(single_sample_fn)(u))

        x = jnp.tanh(u)
        correction = -2.0 * jnp.sum(x * v_u)
        return div_u + correction
    
    def get_divergence(self, model: FlowModel, x_t: jax.Array, t: jax.Array, key: jax.Array, obs: jax.Array = None):
        """
        Public API: Supports both (D,) and (B, D) inputs automatically.
        """
        if x_t.ndim == 1:
            return self._get_single_divergence(model, x_t, t, obs, key)

        # If x_t is (B, D), we need a unique key for each sample in the batch
        keys = jax.random.split(key, x_t.shape[0])
        return jax.vmap(partial(self._get_single_divergence, model))(x_t, t, obs, keys)

    def compute_proxy_entropy_loss(self, key: jax.Array, model: FlowModel, x_noise: jax.Array, x_t: jax.Array, t: jax.Array = None, obs: jax.Array = None) -> jax.Array:
        d = x_noise.shape[-1]
        x_t = jax.lax.stop_gradient(x_t)

        divergence = self.get_divergence(model, x_t, t, key, obs)
        h0_gaussian = 0.5 * d * (1.0 + jnp.log(2 * jnp.pi))
        abs_u = jnp.abs(x_noise)
        correction = jnp.sum(2.0 * (jnp.log(2.0) - abs_u - jax.nn.softplus(-2.0 * abs_u)), axis=-1)
        h_0 = h0_gaussian + jnp.mean(correction)
        proxy_entropy_loss = -h_0 - jnp.mean(divergence)
        return proxy_entropy_loss, h_0+divergence

        # # Compute divergence
        # log_p0 = -0.5 * (d * jnp.log(2 * jnp.pi) + jnp.sum(jnp.square(x_noise), axis=-1))
        # abs_u = jnp.abs(x_noise)
        # correction = jnp.sum(2.0 * (jnp.log(2.0) - abs_u - jax.nn.softplus(-2.0 * abs_u)), axis=-1)
        # log_p0 = log_p0 - correction
        # divergence = self.get_divergence(model, x_t, t, key, obs)
        # divergence = divergence-log_p0
        # proxy_entropy_loss = -jnp.mean(divergence)
        # return proxy_entropy_loss, divergence
    
    def solve_ode_with_log_prob(self, model: FlowModel, shape: Tuple[int, ...], key: jax.Array, obs: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        通过 ODE 求解动作及其精确的对数似然 log p(x)
        """
        k1, k2 = jax.random.split(key)
        u_0 = jax.random.normal(k1, shape)
        
        # 计算初始噪声的 log p0(u0)
        d = shape[-1]
        log_p0 = -0.5 * (d * jnp.log(2 * jnp.pi) + jnp.sum(jnp.square(u_0), axis=-1))
        abs_u = jnp.abs(u_0)
        correction = jnp.sum(2.0 * (jnp.log(2.0) - abs_u - jax.nn.softplus(-2.0 * abs_u)), axis=-1)
        log_p0 = log_p0 - correction

        dt = 1.0 / self.num_timesteps
        
        def body_fn(carry, t_val):
            u, cumulative_div, loop_key = carry
            step_key, next_loop_key = jax.random.split(loop_key)

            t = jnp.full((u.shape[0],), t_val)
            
            # 1. 计算速度场 v
            v = model(t, u, obs)
            
            # 2. 计算当前点的散度 (使用 Hutchinson 估计)
            div = self.get_divergence(model, u, t, step_key, obs) 
            div = jnp.clip(div, -50.0, 50.0)
            
            # 3. 更新状态
            u_next = u + v * dt
            cumulative_div_next = cumulative_div + div * dt
            
            return (u_next, cumulative_div_next, next_loop_key), None

        timesteps = jnp.linspace(0.0, 1.0, self.num_timesteps)
        # 初始累积散度为 0
        initial_div = jnp.zeros((shape[0],))
        (u_final, total_div, _), _ = jax.lax.scan(body_fn, (u_0, initial_div, k2), timesteps)
        
        H_1 = -log_p0 + total_div
        
        x_final = jnp.tanh(u_final)
        return x_final, H_1