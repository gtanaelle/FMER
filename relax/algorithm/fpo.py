from typing import NamedTuple, Tuple, Any
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax
import haiku as hk

from relax.algorithm.base import Algorithm
from relax.network.fpo import FPONet, FPOParams
from relax.utils.typing import Metric

# Extended GAEExperience to carry the rollout-time noise and timestep required by FPO.
class GAEExperience(NamedTuple):
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    done: jax.Array
    next_obs: jax.Array
    ret: jax.Array
    adv: jax.Array
    truncated: jax.Array
    action_info: Any = None  


class FPOOptStates(NamedTuple):
    params: optax.OptState


class FPOTrainState(NamedTuple):
    params: FPOParams
    opt_state: FPOOptStates
    obs_var_sum: jax.Array
    obs_mean: jax.Array
    obs_std: jax.Array
    obs_count: jax.Array
    step: jax.Array


class FPO(Algorithm):

    def __init__(
        self,
        agent: FPONet,
        params: FPOParams,
        learning_rate: float = 3e-4,
        clipping_epsilon: float = 0.05,
        value_loss_coeff: float = 0.25,
        average_losses_before_exp: bool = True,
        normalize_advantage: bool = True,
        max_grad_norm: float = 1.0,
        ratio_log_clip: float = 3.0,
        ratio_value_clip: float = 100.0,
    ):
        self.agent = agent
        self.clipping_epsilon = clipping_epsilon
        self.value_loss_coeff = value_loss_coeff
        self.average_losses_before_exp = average_losses_before_exp
        self.normalize_advantage = normalize_advantage
        self.ratio_log_clip = ratio_log_clip
        self.ratio_value_clip = ratio_value_clip
        self.squash_action_before_step = True

        self.optim = optax.scale_by_adam()

        self.state = FPOTrainState(
            params=params,
            opt_state=FPOOptStates(
                params=self.optim.init(params),
            ),
            obs_var_sum=jnp.zeros((self.agent.obs_dim,), dtype=jnp.float32),
            obs_mean=jnp.zeros((self.agent.obs_dim,), dtype=jnp.float32),
            obs_std=jnp.ones((self.agent.obs_dim,), dtype=jnp.float32),
            obs_count=jnp.zeros((), dtype=jnp.float32),
            step=jnp.int32(0),
        )

        def normalize_obs(obs: jax.Array, mean: jax.Array, std: jax.Array) -> jax.Array:
            return (obs - mean) / std

        def update_obs_stats(
            obs: jax.Array,
            var_sum: jax.Array,
            mean: jax.Array,
            std: jax.Array,
            count: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            del std
            batch_ndims = obs.ndim - mean.ndim
            new_count = count + np.prod(obs.shape[:batch_ndims])
            diff_to_old_mean = obs - mean
            new_mean = mean + jnp.sum(diff_to_old_mean, axis=tuple(range(batch_ndims))) / new_count
            new_var_sum = jnp.sum(
                diff_to_old_mean * (obs - new_mean), axis=tuple(range(batch_ndims))
            )
            var_clipped = jnp.clip(new_var_sum / new_count, 1e-12, 1e12)
            new_std = jnp.sqrt(var_clipped)
            return new_var_sum, new_mean, new_std, new_count

        @jax.jit
        def stateless_update(
            key: jax.Array, state: FPOTrainState, data: GAEExperience
        ) -> Tuple[FPOTrainState, Metric]:
            
            # Extract elements exactly matching your GAEExperience definition
            obs = data.obs 
            action = data.action 
            advantages = data.adv
            returns = data.ret
            action_info = data.action_info 
            truncated = data.truncated

            obs_norm = normalize_obs(obs, state.obs_mean, state.obs_std)

            if self.normalize_advantage:
                adv_mean = jnp.mean(advantages)
                adv_std = jnp.maximum(jnp.std(advantages), 1e-6)
                advantages = (advantages - adv_mean) / adv_std
                advantages = jnp.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)
            
            def loss_fn(params: FPOParams) -> Tuple[jax.Array, dict]:
                # 1. Evaluate Value Function
                value_pred = self.agent.value(params.value, obs_norm)
                v_error = (returns - value_pred) * (1.0 - truncated)
                v_loss = jnp.mean(v_error ** 2) * self.value_loss_coeff
                
                # 2. Re-compute CFM Loss on the training batch
                cfm_loss = self.agent._compute_cfm_loss(
                    params.policy,
                    obs_norm,
                    action,
                    eps=action_info.loss_eps,
                    t=action_info.loss_t
                )
                
                # 3. Calculate Ratio between Old and New Policies using CFM Loss diff
                if self.average_losses_before_exp:
                    loss_diff = (
                        jnp.mean(action_info.initial_cfm_loss, axis=-1, keepdims=True)
                        - jnp.mean(cfm_loss, axis=-1, keepdims=True)
                    )
                    loss_diff = jnp.clip(
                        loss_diff,
                        -self.ratio_log_clip,
                        self.ratio_log_clip,
                    )
                    rho_s = jnp.exp(loss_diff)
                else:
                    loss_diff = jnp.clip(
                        action_info.initial_cfm_loss - cfm_loss,
                        -self.ratio_log_clip,
                        self.ratio_log_clip,
                    )
                    rho_s = jnp.exp(loss_diff)

                rho_s = jnp.nan_to_num(rho_s, nan=1.0, posinf=self.ratio_value_clip, neginf=0.0)
                rho_s = jnp.clip(rho_s, 0.0, self.ratio_value_clip)
                
                # Squeeze to handle broadcasting with advantages if necessary
                if rho_s.shape[-1] == 1 and advantages.ndim == rho_s.ndim - 1:
                    rho_s = jnp.squeeze(rho_s, axis=-1)
                
                # 4. PPO-Style Surrogate Clipping Objective
                surrogate_loss1 = rho_s * advantages
                surrogate_loss2 = (
                    jnp.clip(rho_s, 1.0 - self.clipping_epsilon, 1.0 + self.clipping_epsilon) * advantages
                )
                
                policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))
                policy_loss = jnp.nan_to_num(policy_loss, nan=0.0, posinf=1e6, neginf=-1e6)
                
                total_loss = policy_loss + v_loss
                total_loss = jnp.nan_to_num(total_loss, nan=0.0, posinf=1e6, neginf=-1e6)
                
                metrics = {
                    "policy_loss": policy_loss,
                    "value_loss": v_loss,
                    "policy_ratio_mean": jnp.mean(rho_s),
                    "policy_ratio_max": jnp.max(rho_s),
                    "policy_ratio_min": jnp.min(rho_s),
                    "advantages_mean": jnp.mean(advantages),
                    "value_mean": jnp.mean(value_pred)
                }
                
                return total_loss, metrics

            # Compute Gradients
            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

            # Optimization Update
            param_update, new_opt_state = self.optim.update(grads, state.opt_state.params)
            param_update = jax.tree.map(lambda x: -learning_rate * x, param_update)
            new_params = optax.apply_updates(state.params, param_update)

            new_state = FPOTrainState(
                params=new_params,
                opt_state=FPOOptStates(params=new_opt_state),
                obs_var_sum=state.obs_var_sum,  # Keep original state
                obs_mean=state.obs_mean,        # Keep original state
                obs_std=state.obs_std,          # Keep original state
                obs_count=state.obs_count,      # Keep original state
                step=state.step + 1
            )
            
            return new_state, metrics

        # --- JIT-compiled step and get_value functions for rollouts ---
        @jax.jit
        def stateless_step(key: jax.Array, state: FPOTrainState, obs: jax.Array):
            obs_norm = normalize_obs(obs, state.obs_mean, state.obs_std)
            action, action_info = self.agent.sample_action(
                key, state.params.policy, obs_norm, deterministic=False
            )
            value = self.agent.value(state.params.value, obs_norm)
            return action, value, action_info

        @jax.jit
        def stateless_get_value(state: FPOTrainState, obs: jax.Array):
            obs_norm = normalize_obs(obs, state.obs_mean, state.obs_std)
            return self.agent.value(state.params.value, obs_norm)

        # Bind the step function internally
        self._stateless_step = stateless_step
        self._stateless_get_value = stateless_get_value

        # Register with common behavior, injecting the value function so get_value() works
        self._implement_common_behavior(
            stateless_update, 
            self.agent.get_action, 
            self.agent.get_deterministic_action,
        )

    def get_policy_params(self):
        return self.state.params.policy

    def save_policy(self, path: str) -> None:
        payload = {
            "policy": jax.device_get(self.get_policy_params()),
            "obs_mean": jax.device_get(self.state.obs_mean),
            "obs_std": jax.device_get(self.state.obs_std),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def step(self, key: jax.Array, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Any]:
        """Runs policy and value networks for rollout collection."""
        action, value, action_info = self._stateless_step(key, self.state, obs)
        # Convert output back to numpy arrays for the Gym environment
        return np.asarray(action), np.asarray(value), action_info

    def get_value(self, obs: np.ndarray) -> np.ndarray:
        value = self._stateless_get_value(self.state, obs)
        return np.asarray(value)
    
    def update_obs_stats(self, obs_batch: np.ndarray):
        """Updates observation running statistics exactly once per rollout."""
        self.state = _jit_update_obs_stats(self.state, obs_batch)
    
# Place this OUTSIDE your FPO class
@jax.jit
def _jit_update_obs_stats(state: FPOTrainState, obs: jax.Array) -> FPOTrainState:
    batch_ndims = obs.ndim - state.obs_mean.ndim
    new_count = state.obs_count + jnp.prod(jnp.array(obs.shape[:batch_ndims]))
    
    diff_to_old_mean = obs - state.obs_mean
    new_mean = state.obs_mean + jnp.sum(diff_to_old_mean, axis=tuple(range(batch_ndims))) / new_count
    
    new_var_sum = state.obs_var_sum + jnp.sum(
        diff_to_old_mean * (obs - new_mean), axis=tuple(range(batch_ndims))
    )
    
    var_clipped = jnp.clip(new_var_sum / new_count, 1e-12, 1e12)
    new_std = jnp.sqrt(var_clipped)
    
    return FPOTrainState(
        params=state.params,
        opt_state=state.opt_state,
        obs_var_sum=new_var_sum,
        obs_mean=new_mean,
        obs_std=new_std,
        obs_count=new_count,
        step=state.step
    )