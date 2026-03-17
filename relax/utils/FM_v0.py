from typing import Protocol, Tuple, Optional
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
    lambda_: float = 0.1
    action_dim: int = 1
    include_entropy: bool = True
    use_accurate_entropy: bool = False

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
        
        return jnp.clip(x_final, -1.0, 1.0)

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

    def compute_loss(self, sigma_min: jax.Array, key: jax.Array, model: FlowModel, q_weights: jax.Array, x_start: jax.Array,  log_alpha: jax.Array, obs: jax.Array) -> jax.Array:
        x_start = jax.lax.stop_gradient(x_start)
        q_weights = jax.lax.stop_gradient(q_weights)
        log_alpha = jax.lax.stop_gradient(log_alpha)

        batch_size = x_start.shape[0]
        k1, k2, k3, k4 = jax.random.split(key, 4)
        
        # Sample time t ~ U(0, 1)
        t = jax.random.uniform(k1, (batch_size,))
        x_noise = jax.random.normal(k2, x_start.shape)
        x_t = self.get_xt(sigma_min, x_start, x_noise, t)
        cfm_loss = self.compute_cfm_loss(sigma_min, model, t, x_t, x_start, x_noise, q_weights, obs)

        if not self.include_entropy:
            return cfm_loss, None
        else:
            entropy_loss = self.compute_proxy_entropy_loss(k3, model, x_noise, x_t, t, obs)
            random_alpha = jax.random.normal(k4, log_alpha.shape)*self.lambda_
            total_policy_loss = cfm_loss +jnp.exp(log_alpha)*entropy_loss
            return total_policy_loss, cfm_loss, -entropy_loss
    
    # def get_divergence(self, model: FlowModel, x_t: jax.Array, t: jax.Array, key: jax.Array):
    #     """
    #     Hutchinson proxy_entropy_loss
    #     """
    #     def v_fn(x):
    #         return model(t, x)
        
    #     eps = jax.random.normal(key, x_t.shape)
    #     _, jvp_val = jax.jvp(v_fn, (x_t,), (eps,))
    #     divergence = jnp.sum(jvp_val * eps, axis=-1)
    #     return divergence
    
    def _get_single_divergence(self, model: FlowModel, x: jax.Array, t: jax.Array, obs: jax.Array) -> jax.Array:
        """
        Input: x shape (D,), t shape ()
        Output: scalar ()
        """
        def single_sample_fn(x_in):
            # model expects t: (B,), x: (B, D)
            return model(t[None], x_in[None, :], obs[None, :])[0]
        return jnp.trace(jax.jacrev(single_sample_fn)(x))
    
    def get_divergence(self, model: FlowModel, x_t: jax.Array, t: jax.Array, key: jax.Array, obs: jax.Array = None):
        """
        Public API: Supports both (D,) and (B, D) inputs automatically.
        """
        if x_t.ndim == 1:
            return self._get_single_divergence(model, x_t, t, obs)

        return jax.vmap(partial(self._get_single_divergence, model))(x_t, t, obs)

    def compute_proxy_entropy_loss(self, key: jax.Array, model: FlowModel, x_noise: jax.Array, x_t: jax.Array, t: jax.Array = None, obs: jax.Array = None) -> jax.Array:
        d = x_noise.shape[-1]
        log_p0 = -0.5 * (d * jnp.log(2 * jnp.pi) + jnp.sum(jnp.square(x_noise), axis=-1))
        x_t = jax.lax.stop_gradient(x_t)

        # Compute divergence
        divergence = self.get_divergence(model, x_t, t, key, obs)

        # divergence = 5.0 * jnp.tanh(divergence / 5.0)
        divergence = divergence-log_p0
        
        # Proxy entropy loss is negative divergence
        proxy_entropy_loss = -jnp.mean(divergence)
        return proxy_entropy_loss
    
    def solve_ode_with_log_prob(self, model: FlowModel, shape: Tuple[int, ...], key: jax.Array, obs: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        通过 ODE 求解动作及其精确的对数似然 log p(x)
        """
        k1, k2 = jax.random.split(key)
        x_0 = jax.random.normal(k1, shape)
        
        # 计算初始噪声的 log p0(x0)
        d = shape[-1]
        log_p0 = -0.5 * (d * jnp.log(2 * jnp.pi) + jnp.sum(jnp.square(x_0), axis=-1))
        
        dt = 1.0 / self.num_timesteps
        
        def body_fn(carry, t_val):
            x, cumulative_div = carry
            t = jnp.full((x.shape[0],), t_val)
            
            # 1. 计算速度场 v
            v = model(t, x, obs)
            
            # 2. 计算当前点的散度 (使用 Hutchinson 估计)
            # 注意：在生成动作时，我们通常需要这里的梯度
            div = self.get_divergence(model, x, t, k2, obs) 
            div = jnp.clip(div, -50.0, 50.0)
            
            # 3. 更新状态
            x_next = x + v * dt
            cumulative_div_next = cumulative_div + div * dt
            
            return (x_next, cumulative_div_next), None

        timesteps = jnp.linspace(0.0, 1.0, self.num_timesteps)
        # 初始累积散度为 0
        initial_div = jnp.zeros((shape[0],))
        (x_final, total_div), _ = jax.lax.scan(body_fn, (x_0, initial_div), timesteps)
        
        # log p1(x1) = log p0(x0) - total_div
        log_p1 = log_p0 - total_div
        
        print(x_final.shape, log_p1.shape)
        return jnp.clip(x_final, -1.0, 1.0), -log_p1