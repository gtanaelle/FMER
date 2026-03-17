from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple, Any

import jax
import jax.numpy as jnp
import haiku as hk

from relax.network.blocks import Activation, ValueNet
from relax.utils.jax_utils import random_key_from_data


class FPOParams(NamedTuple):
    policy: hk.Params
    value: hk.Params

class FPOActionInfo(NamedTuple):
    loss_eps: jax.Array
    loss_t: jax.Array
    initial_cfm_loss: jax.Array


@dataclass
class FPONet:
    policy: Callable[[hk.Params, jax.Array, jax.Array, jax.Array], jax.Array]
    value: Callable[[hk.Params, jax.Array], jax.Array]
    obs_dim: int
    flow_steps: int
    act_dim: int
    timestep_embed_dim: int
    policy_mlp_output_scale: float
    feather_std: float
    output_mode: str
    n_samples_per_action: int
    discretize_t_for_training: bool

    def embed_timestep(self, t: jax.Array) -> jax.Array:
        freqs = 2 ** jnp.arange(self.timestep_embed_dim // 2)
        scaled_t = t * freqs
        out = jnp.concatenate([jnp.cos(scaled_t), jnp.sin(scaled_t)], axis=-1)
        return out

    def _compute_cfm_loss(
        self,
        policy_params: hk.Params,
        obs: jax.Array,
        action: jax.Array,
        eps: jax.Array,
        t: jax.Array,
    ) -> jax.Array:
        batch_dims = action.shape[:-1]
        samples_dim = self.n_samples_per_action
        obs_dim = obs.shape[-1]
        sample_shape = (*batch_dims, samples_dim)

        x_t = t * eps + (1.0 - t) * action[..., None, :]
        obs_expanded = jnp.broadcast_to(obs[..., None, :], (*sample_shape, obs_dim))
        t_embed = self.embed_timestep(t)

        network_pred = self.policy(policy_params, obs_expanded, x_t, t_embed) * self.policy_mlp_output_scale

        if self.output_mode == "u":
            velocity_gt = eps - action[..., None, :]
            out = jnp.mean((network_pred - velocity_gt) ** 2, axis=-1)
        elif self.output_mode == "u_but_supervise_as_eps":
            velocity_pred = network_pred
            x0_pred = x_t - t * velocity_pred
            x1_pred = x0_pred + velocity_pred
            out = jnp.mean((eps - x1_pred) ** 2, axis=-1)
        else:
            raise ValueError(f"Unknown output_mode: {self.output_mode}")
            
        return out

    def sample_action(
        self, key: jax.Array, policy_params: hk.Params, obs: jax.Array, deterministic: bool = False
    ) -> Tuple[jax.Array, FPOActionInfo]:
        batch_dims = obs.shape[:-1]

        def euler_step(carry, inputs):
            x_t = carry
            t_current, t_next = inputs
            dt = t_next - t_current

            t_embed = jnp.broadcast_to(self.embed_timestep(t_current[None]), (*batch_dims, self.timestep_embed_dim))
            velocity = self.policy(policy_params, obs, x_t, t_embed) * self.policy_mlp_output_scale
            x_t_next = x_t + dt * velocity
            return x_t_next, x_t

        prng_sample, prng_loss, prng_feather, prng_noise = jax.random.split(key, 4)

        full_t_path = jnp.linspace(1.0, 0.0, self.flow_steps + 1)
        t_current = full_t_path[:-1]
        t_next = full_t_path[1:]

        x0, _ = jax.lax.scan(
            euler_step,
            init=jax.random.normal(prng_sample, (*batch_dims, self.act_dim)),
            xs=(t_current, t_next)
        )

        if not deterministic:
            perturb = jax.random.normal(prng_feather, (*batch_dims, self.act_dim)) * self.feather_std
            x0 = x0 + perturb

        sample_shape = (*batch_dims, self.n_samples_per_action)
        prng_eps, prng_t = jax.random.split(prng_loss)
        eps = jax.random.normal(prng_eps, (*sample_shape, self.act_dim))

        if self.discretize_t_for_training:
            t_idx = jax.random.randint(prng_t, shape=(*sample_shape, 1), minval=0, maxval=self.flow_steps)
            t = t_current[t_idx]
        else:
            t = jax.random.uniform(prng_t, (*sample_shape, 1))

        initial_cfm_loss = self._compute_cfm_loss(policy_params, obs, x0, eps, t)

        return x0, FPOActionInfo(loss_eps=eps, loss_t=t, initial_cfm_loss=initial_cfm_loss)

    def get_action(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        action, _ = self.sample_action(key, policy_params, obs, deterministic=False)
        return action

    def get_deterministic_action(self, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        key = random_key_from_data(obs)
        action, _ = self.sample_action(key, policy_params, obs, deterministic=True)
        return action



def create_fpo_net(
    key: jax.Array,
    obs_dim: int,
    act_dim: int,
    policy_hidden_sizes: Sequence[int] = (32, 32, 32, 32),
    hidden_sizes: Sequence[int] = (256, 256, 256, 256, 256),
    timestep_embed_dim: int = 8,
    flow_steps: int = 10,
    policy_mlp_output_scale: float = 0.25,
    feather_std: float = 0.0,
    output_mode: str = "u_but_supervise_as_eps",
    n_samples_per_action: int = 8,
    discretize_t_for_training: bool = True,
    activation: Activation = jax.nn.silu,
) -> Tuple[FPONet, FPOParams]:

    value = hk.without_apply_rng(hk.transform(lambda obs: ValueNet(hidden_sizes, activation)(obs)))

    def policy_fn(obs, act, t_embed):
        from relax.network.blocks import mlp
        x = jnp.concatenate([obs, act, t_embed], axis=-1)
        return mlp(policy_hidden_sizes, act_dim, activation, lambda x: x)(x)

    policy = hk.without_apply_rng(hk.transform(policy_fn))
    

    @jax.jit
    def init(key, obs, act, t_embed):
        policy_key, value_key = jax.random.split(key)
        policy_params = policy.init(policy_key, obs, act, t_embed)
        value_params = value.init(value_key, obs)
        return FPOParams(policy_params, value_params)

    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    sample_t_embed = jnp.zeros((1, timestep_embed_dim))
    params = init(key, sample_obs, sample_act, sample_t_embed)

    net = FPONet(
        policy=policy.apply,
        value=value.apply,
        obs_dim=obs_dim,
        flow_steps=flow_steps,
        act_dim=act_dim,
        timestep_embed_dim=timestep_embed_dim,
        policy_mlp_output_scale=policy_mlp_output_scale,
        feather_std=feather_std,
        output_mode=output_mode,
        n_samples_per_action=n_samples_per_action,
        discretize_t_for_training=discretize_t_for_training,
    )
    return net, params