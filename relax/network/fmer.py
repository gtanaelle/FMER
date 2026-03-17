from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple, Union

import jax, jax.numpy as jnp
import haiku as hk
import math

from relax.network.blocks import Activation, FMERPolicyNet, QNet, DistributionalQNet2
from relax.utils.FM import FlowMatching
from relax.utils.SM import ScoreMatching
from relax.utils.jax_utils import random_key_from_data

class FMERParams(NamedTuple):
    q1: hk.Params
    q2: hk.Params
    target_q1: hk.Params
    target_q2: hk.Params
    policy: hk.Params
    target_policy: hk.Params
    log_alpha: jax.Array
    log_tau: jax.Array


@dataclass
class FMERNet:
    q: Callable[[hk.Params, jax.Array, jax.Array], jax.Array]
    policy: Callable[[hk.Params, jax.Array, jax.Array, jax.Array], jax.Array]
    num_timesteps: int
    act_dim: int
    num_particles: int
    target_entropy: float
    tau: float = 0.1

    @property
    def flow_matching(self) -> FlowMatching:
        return FlowMatching(self.num_timesteps, action_dim=self.act_dim)

    # @property
    # def score_matching(self) -> ScoreMatching:
    #     return ScoreMatching(self.num_timesteps, action_dim=self.act_dim)

    def get_action(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array, sigma_t: float=None, num_particles: int=0) -> jax.Array:
        if num_particles == 0:
            num_particles = self.num_particles
        policy_params, q1_params, q2_params, _ = policy_params

        def model_fn(t, x, obs):
            return self.policy(policy_params, obs, x, t)

        def sample(key: jax.Array) -> Union[jax.Array, jax.Array]:
            act = self.flow_matching.solve_ode(model_fn, (*obs.shape[:-1], self.act_dim), key, obs)

            q1 = self.q(q1_params, obs, act)
            q2 = self.q(q2_params, obs, act)
            q = jnp.minimum(q1, q2)
            return act, q

        # key, noise_key = jax.random.split(key)
        if num_particles == 1:
            act = sample(key)
        else:
            keys = jax.random.split(key, num_particles)
            acts, qs = jax.vmap(sample)(keys)

            v = jnp.mean(qs, axis=0)
            q = jnp.max(qs, axis=0)
            q_best_ind = jnp.argmax(qs, axis=0, keepdims=True)
            act = jnp.take_along_axis(acts, q_best_ind[..., None], axis=0).squeeze(axis=0)
        return act, q, v
       
    def get_action_for_training(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array, sigma_t: float=None,  num_particles: int=0) -> jax.Array:
        policy_params, q1_params, q2_params, log_tau = policy_params

        def model_fn(t, x, obs):
            return self.policy(policy_params, obs, x, t)

        def sample(key: jax.Array) -> Union[jax.Array, jax.Array]:
            act, ent = self.flow_matching.solve_ode_with_log_prob(model_fn, (*obs.shape[:-1], self.act_dim), key, obs)

            q1 = self.q(q1_params, obs, act)
            q2 = self.q(q2_params, obs, act)
            q = jnp.minimum(q1, q2)
            return act, q, ent

        # key, noise_key = jax.random.split(key)
        if num_particles == 1:
            act = sample(key)
        else:
            keys = jax.random.split(key, num_particles)
            acts, qs, ent = jax.vmap(sample)(keys)
            ent = jnp.mean(ent, axis=0)

            # default weights
            baseline = jnp.mean(qs, axis=0, keepdims=True)
            advantages = qs - baseline
            adv_shift = advantages - jnp.max(advantages, axis=0, keepdims=True)
            weights = jax.nn.softmax(adv_shift/jnp.exp(log_tau), axis=0)
            # adv_std = jnp.std(advantages, axis=0, keepdims=True)
            # weights = jax.nn.softmax(advantages/(jnp.exp(log_tau)*adv_std+1e-8), axis=0)
            ess = 1.0 / jnp.sum(jnp.square(weights), axis=0)
            return acts, weights, ent, jnp.mean(ess)

    def get_batch_actions(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array, q_func: Callable) -> jax.Array:
        batch_flatten_obs = obs.repeat(self.num_particles, axis=0)
        batch_flatten_actions = self.get_action(key, policy_params, batch_flatten_obs)
        batch_q = q_func(batch_flatten_obs, batch_flatten_actions).reshape(-1, self.num_particles)
        max_q_idx = batch_q.argmax(axis=1)
        batch_action = batch_flatten_actions.reshape(obs.shape[0], -1, self.act_dim) # ?
        slice = lambda x, y: x[y]
        # action: batch_size, repeat_size, idx: batch_size
        best_action = jax.vmap(slice, (0, 0))(batch_action, max_q_idx)
        return best_action

    def get_deterministic_action(self, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        key = random_key_from_data(obs)
        policy_params, q1_params, q2_params, log_tau = policy_params
        policy_params = (policy_params, q1_params, q2_params, log_tau)
        action, _, _ = self.get_action(key, policy_params, obs)
        return action

    def q_evaluate(
        self, key: jax.Array, q_params: hk.Params, obs: jax.Array, act: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        q_mean, q_std = self.q(q_params, obs, act)
        z = jax.random.normal(key, q_mean.shape)
        z = jnp.clip(z, -3.0, 3.0)  # NOTE: Why not truncated normal?
        q_value = q_mean + q_std * z
        return q_mean, q_std, q_value

def create_fmer_net(
    key: jax.Array,
    obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int],
    diffusion_hidden_sizes: Sequence[int],
    activation: Activation = jax.nn.relu,
    num_timesteps: int = 20,
    num_particles: int = 4,
    tau: float = 0.01,
    ) -> Tuple[FMERNet, FMERParams]:
    q = hk.without_apply_rng(hk.transform(lambda obs, act: QNet(hidden_sizes, activation)(obs, act)))
    # q = hk.without_apply_rng(hk.transform(lambda obs, act: DistributionalQNet2(hidden_sizes, activation)(obs, act)))
    policy = hk.without_apply_rng(hk.transform(lambda obs, act, t: FMERPolicyNet(diffusion_hidden_sizes, activation)(obs, act, t)))

    @jax.jit
    def init(key, obs, act):
        q1_key, q2_key, policy_key = jax.random.split(key, 3)
        q1_params = q.init(q1_key, obs, act)
        q2_params = q.init(q2_key, obs, act)
        target_q1_params = q1_params
        target_q2_params = q2_params
        policy_params = policy.init(policy_key, obs, act, 0)
        target_policy_params = policy_params
        log_alpha = jnp.array(1.0, dtype=jnp.float32)
        log_tau = jnp.array(jnp.log(tau), dtype=jnp.float32)
        # log_alpha = jnp.array(0.01, dtype=jnp.float32) #jnp.array(math.log(1), dtype=jnp.float32) # math.log(3) or math.log(5) choose one
        return FMERParams(q1_params, q2_params, target_q1_params, target_q2_params, policy_params, target_policy_params, log_alpha, log_tau)

    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    params = init(key, sample_obs, sample_act)

    net = FMERNet(q=q.apply, policy=policy.apply, num_timesteps=num_timesteps, act_dim=act_dim,
                    target_entropy=-act_dim*0.9, num_particles=num_particles, tau=tau)
    return net, params
