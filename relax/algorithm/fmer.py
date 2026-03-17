from typing import NamedTuple, Tuple

import jax, jax.numpy as jnp
import numpy as np
import optax
import haiku as hk
import pickle

from relax.algorithm.base import Algorithm
from relax.network.fmer import FMERNet, FMERParams
from relax.utils.experience import Experience
from relax.utils.typing import Metric

# --- New: Running Statistics Container ---
class RunningMeanStd(NamedTuple):
    mean: jax.Array
    var: jax.Array
    count: jax.Array

# Optimization states 
class FMEROptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState
    policy: optax.OptState
    log_alpha: optax.OptState
    log_tau: optax.OptState

# Training state
class FMERTrainState(NamedTuple):
    params: FMERParams
    opt_state: FMEROptStates
    step: int
    entropy: float
    reward_stats: RunningMeanStd  # Added reward statistics to state

class FMER(Algorithm):
    def __init__(
        self,
        agent: FMERNet,
        params: FMERParams,
        *,
        gamma: float = 0.99,
        lr: float = 1e-4,
        lr_schedule_end = 5e-5,
        alpha_lr: float = 3e-4,
        tau_lr: float = 3e-4,
        tau: float = 0.005,
        delay_update: int = 2,
        reward_scale: float = 0.2,
        num_samples: int = 200,
    ):
        self.agent = agent
        self.gamma = gamma
        self.tau = tau
        self.delay_update = delay_update
        self.reward_scale = reward_scale
        self.num_samples = num_samples
        self.optim = optax.adam(lr)
        lr_schedule = optax.schedules.linear_schedule(
            init_value=lr,
            end_value=lr_schedule_end,
            transition_steps=int(5e4),
            transition_begin=int(2.5e4),
        )
        self.policy_optim = optax.adam(learning_rate=lr_schedule)
        self.alpha_optim = optax.adam(alpha_lr)
        self.tau_optim = optax.adam(tau_lr)
        self.entropy = 0.0
        self.target_entropy = -float(agent.act_dim)
        self.target_ess = 3# float(agent.num_particles) / 3.0  # Target Effective Sample Size

        # Initialize running reward statistics
        initial_reward_stats = RunningMeanStd(
            mean=jnp.array(0.0),
            var=jnp.array(1.0),
            count=jnp.array(1e-4),
        )

        self.state = FMERTrainState(
            params=params,
            opt_state=FMEROptStates(
                q1=self.optim.init(params.q1),
                q2=self.optim.init(params.q2),
                policy=self.policy_optim.init(params.policy),
                log_alpha=self.alpha_optim.init(params.log_alpha),
                log_tau=self.alpha_optim.init(params.log_tau),
            ),
            step=jnp.int32(0),
            entropy=jnp.float32(0.0),
            reward_stats=initial_reward_stats,
        )

        # Helper function for Parallel Welford's Algorithm
        def update_running_stats(stats: RunningMeanStd, batch_vals: jax.Array) -> RunningMeanStd:
            batch_mean = jnp.mean(batch_vals)
            batch_var = jnp.var(batch_vals)
            batch_count = batch_vals.shape[0]

            delta = batch_mean - stats.mean
            tot_count = stats.count + batch_count

            new_mean = stats.mean + delta * batch_count / tot_count
            m_a = stats.var * stats.count
            m_b = batch_var * batch_count
            m_2 = m_a + m_b + jnp.square(delta) * stats.count * batch_count / tot_count
            new_var = m_2 / tot_count
            
            return RunningMeanStd(mean=new_mean, var=new_var, count=tot_count)

        @jax.jit
        def stateless_update(
            key: jax.Array, state: FMERTrainState, data: Experience
        ) -> Tuple[FMEROptStates, Metric]:
            # update networks
            def param_update(optim, params, grads, opt_state):
                update, new_opt_state = optim.update(grads, opt_state)
                new_params = optax.apply_updates(params, update)
                return new_params, new_opt_state

            def delay_param_update(optim, params, grads, opt_state):
                return jax.lax.cond(
                    step % self.delay_update == 0,
                    lambda params, opt_state: param_update(optim, params, grads, opt_state),
                    lambda params, opt_state: (params, opt_state),
                    params, opt_state
                )
            
            def delay_target_update(params, target_params, tau):
                return jax.lax.cond(
                    step % self.delay_update == 0,
                    lambda target_params: optax.incremental_update(params, target_params, tau),
                    lambda target_params: target_params,
                    target_params
                )

            def _exp_target_entropy(step: jax.Array, start: int = 10_000, end: int = 200_000,
                                    sharpness: float = 5.0) -> jax.Array:
                # t in [0,1]
                t = jnp.clip((step - start) / (end - start), 0.0, 1.0)
                # exponential rise from 0 to 1
                frac = (1.0 - jnp.exp(-sharpness * t)) / (1.0 - jnp.exp(-sharpness))
                return -float(agent.act_dim)*frac

            obs, action, reward, next_obs, done = data.obs, data.action, data.reward, data.next_obs, data.done
            
            # reward *= self.reward_scale
            # --- START Running Reward Normalization ---
            new_reward_stats = update_running_stats(state.reward_stats, reward)            
            # Normalize rewards (divide by std, do NOT subtract mean for rewards)
            reward = reward / (jnp.sqrt(new_reward_stats.var) + 1e-8)
            # clamp extreme values:
            reward = jnp.clip(reward, -10.0, 10.0)
            
            q1_params, q2_params, target_q1_params, target_q2_params, policy_params, target_policy_params, log_alpha, log_tau = state.params
            q1_opt_state, q2_opt_state, policy_opt_state, log_alpha_opt_state, log_tau_opt_state = state.opt_state
            step = state.step
            next_eval_key, new_eval_key, new_q1_eval_key, new_q2_eval_key, diffusion_noise_key = jax.random.split(
                key, 5)
            

            next_action, _, _ = self.agent.get_action(next_eval_key, (policy_params, q1_params, q2_params, log_tau), next_obs, num_particles=8)
            q1_target = self.agent.q(target_q1_params, next_obs, next_action)
            q2_target = self.agent.q(target_q2_params, next_obs, next_action)
            q_target = jnp.minimum(q1_target, q2_target)  # - jnp.exp(log_alpha) * next_logp
            q_backup = reward + (1 - done) * self.gamma * q_target

            def q_loss_fn(q_params: hk.Params) -> jax.Array:
                q = self.agent.q(q_params, obs, action)
                q_loss = jnp.mean((q - q_backup) ** 2)
                return q_loss, q

            (q1_loss, q1), q1_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q1_params)
            (q2_loss, q2), q2_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q2_params)
            q1_grads = jax.tree.map(lambda g: jnp.clip(g, -50.0, 50.0), q1_grads)
            q2_grads = jax.tree.map(lambda g: jnp.clip(g, -50.0, 50.0), q2_grads)
            q1_grads, _ = optax.clip_by_global_norm(10.0).update(q1_grads, None)
            q2_grads, _ = optax.clip_by_global_norm(10.0).update(q2_grads, None)
            q1_params, q1_opt_state = param_update(self.optim, q1_params, q1_grads, q1_opt_state)
            q2_params, q2_opt_state = param_update(self.optim, q2_params, q2_grads, q2_opt_state)

            # update policy
            def policy_loss_fn(policy_params, q1_params, q2_params, log_alpha, obs) -> jax.Array:
                new_acts, softmax_weights, entropy, ess = self.agent.get_action_for_training(new_eval_key, (policy_params, target_q1_params, target_q2_params, log_tau), obs, num_particles=8)

                def denoiser(t, x, obs):
                    return self.agent.policy(policy_params, obs, x, t)

                entropy = jnp.tile(entropy, (new_acts.shape[0], 1))
                obs = jnp.tile(obs, (new_acts.shape[0], 1, 1))
                q_weights = jnp.reshape(softmax_weights, (-1,))
                new_acts = jnp.reshape(new_acts, (-1, self.agent.act_dim))
                entropy = jnp.reshape(entropy, (-1,))
                obs = jnp.reshape(obs, (-1, obs.shape[-1]))

                total_policy_loss, cfm_loss, div_val = self.agent.flow_matching.compute_loss(
                    0.0, #sigma_min
                    diffusion_noise_key,
                    denoiser,
                    q_weights,
                    jax.lax.stop_gradient(new_acts),
                    log_alpha,
                    jax.lax.stop_gradient(obs)
                )

                return total_policy_loss, (q_weights, cfm_loss, div_val, entropy, new_acts, ess)
            
            (total_loss, (q_weights, cfm_loss, div_val, entropy, new_acts, ess)), policy_grads \
                = jax.value_and_grad(policy_loss_fn, has_aux=True)(policy_params, q1_params, q2_params, log_alpha, obs)
            policy_grads, _ = optax.clip_by_global_norm(2.0).update(policy_grads, None)
            policy_params, policy_opt_state = delay_param_update(self.policy_optim, policy_params, policy_grads, policy_opt_state)

            # update alpha
            def alpha_loss_fn(log_alpha, ent_val):
                ent_val = jax.lax.stop_gradient(ent_val)
                target_entropy = _exp_target_entropy(step)
                alpha_loss =  log_alpha * (jnp.mean(ent_val) - target_entropy) #jnp.exp(log_alpha)*( env - self.target_entropy)
                return alpha_loss
            
            alpha_grads = jax.grad(alpha_loss_fn)(log_alpha, entropy)
            log_alpha, log_alpha_opt_state = param_update(self.alpha_optim, log_alpha, alpha_grads, log_alpha_opt_state)

            # update tau
            def tau_loss_fn(log_tau, ess_val):
                ess_val = jax.lax.stop_gradient(ess_val)
                tau_loss =  log_tau * (jnp.mean(ess_val) - self.target_ess) #jnp.exp(log_tau)*( self.target_ess - jnp.mean(ess_val))
                return tau_loss
            
            tau_grads = jax.grad(tau_loss_fn)(log_tau, ess)
            log_tau, log_tau_opt_state = param_update(self.tau_optim, log_tau, tau_grads, log_tau_opt_state)
            # log_tau = jnp.clip(log_tau, jnp.log(0.1), jnp.log(5.0))  # Clip log_tau to prevent extreme values

            target_q1_params = delay_target_update(q1_params, target_q1_params, self.tau)
            target_q2_params = delay_target_update(q2_params, target_q2_params, self.tau)
            target_policy_params = delay_target_update(policy_params, target_policy_params, self.tau)

            state = FMERTrainState(
                params=FMERParams(q1_params, q2_params, target_q1_params, target_q2_params, policy_params, target_policy_params, log_alpha, log_tau),
                opt_state=FMEROptStates(q1=q1_opt_state, q2=q2_opt_state, policy=policy_opt_state, log_alpha=log_alpha_opt_state, log_tau=log_tau_opt_state),
                step=step + 1,
                entropy=jnp.float32(0.0),
                reward_stats=new_reward_stats, # Pass updated reward stats
            )
            info = {
                "q1_loss": q1_loss,
                "q2_loss": q2_loss,
                "divergence": jnp.mean(div_val),
                "divergence_max": jnp.max(div_val),
                "divergence_min": jnp.min(div_val),
                "divergence_std": jnp.std(div_val),
                "entropy": jnp.mean(entropy),
                "target_entropy": _exp_target_entropy(step),
                "alpha": jnp.exp(log_alpha),
                "tau": jnp.exp(log_tau),
                "ess": ess,
                "cfm_loss": cfm_loss,
                "policy_loss": total_loss,
                "q1_mean": jnp.mean(q1),
                "q1_std": jnp.std(q1),
                "q2_mean": jnp.mean(q2),
                "q2_std": jnp.std(q2),
                "effect_reward": jnp.mean(reward),
                "effect_reward_std": jnp.std(reward),
                # "reward_scaler_var": new_reward_stats.var, # Monitor reward variance
                # "reward_scaler_mean": new_reward_stats.mean, # Monitor reward mean
                "policy_grads_max": jnp.max(jnp.array(jax.tree_util.tree_leaves(jax.tree.map(lambda x: jnp.max(jnp.abs(x)), policy_grads)))),
                "policy_grads_mean": jnp.mean(jnp.array(jax.tree_util.tree_leaves(jax.tree.map(lambda x: jnp.mean(jnp.abs(x)), policy_grads)))),
                "critic_grads_max": jnp.max(jnp.array(jax.tree_util.tree_leaves(jax.tree.map(lambda x: jnp.max(jnp.abs(x)), q1_grads)))),
                "critic_grads_mean": jnp.mean(jnp.array(jax.tree_util.tree_leaves(jax.tree.map(lambda x: jnp.mean(jnp.abs(x)), q1_grads)))),
                "action_mean": jnp.mean(new_acts),
                "action_std": jnp.std(new_acts),
                "action_min": jnp.min(new_acts),
                "action_max": jnp.max(new_acts),
                "weights_std": jnp.std(q_weights),
                "weights_mean": jnp.mean(q_weights),
                "weights_min": jnp.min(q_weights),
                "weights_max": jnp.max(q_weights),
            }
            return state, info

        self._implement_common_behavior(stateless_update, self.agent.get_action, self.agent.get_deterministic_action)

    def get_policy_params(self):
        return (self.state.params.policy, self.state.params.q1, self.state.params.q2, self.state.params.log_tau) 

    def get_policy_params_to_save(self):
        return (self.state.params.target_policy, self.state.params.target_q1, self.state.params.target_q2, self.state.params.log_tau)

    def save_policy(self, path: str) -> None:
        policy = jax.device_get(self.get_policy_params_to_save())
        with open(path, "wb") as f:
            pickle.dump(policy, f)

    def get_action(self, key: jax.Array, obs: np.ndarray) -> np.ndarray:
        action, _, _ = self._get_action(key, self.get_policy_params_to_save(), obs)
        return np.asarray(action)
