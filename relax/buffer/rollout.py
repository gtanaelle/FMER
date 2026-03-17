from typing import Any, Generator
import numpy as np
import jax

# Assuming you put the GAEExperience definition in relax.utils.experience
from relax.utils.experience import GAEExperience

class RolloutBuffer:
    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95, seed: int = 0):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.action_infos = []
        self.truncated = []
        self.returns = None
        self.advantages = None

    def add(
        self, 
        obs: np.ndarray, 
        action: np.ndarray, 
        reward: np.ndarray, 
        value: np.ndarray, 
        terminated: np.ndarray, 
        truncated: np.ndarray,  
        action_info: Any
    ):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(terminated)     # Stores ONLY terminated
        self.truncated.append(truncated)  # Stores ONLY truncated
        self.action_infos.append(action_info)

    def compute_returns_and_advantages(self, last_value: np.ndarray):
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        terminateds = np.array(self.dones)
        truncateds = np.array(self.truncated)
        
        # Bootstrap values for the next state (shifting is correct here)
        next_values = np.append(values[1:], [last_value], axis=0)
        
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae_lam = 0
        
        # Notice we do NOT shift `terminateds` or `truncateds`. 
        # `terminateds[step]` perfectly aligns with `next_values[step]`.
        for step in reversed(range(len(rewards))):
            # Mask truncated transitions in both TD errors and the GAE recursion.
            # This avoids bootstrapping through auto-reset observations.
            # Corrected GAE logic
            next_non_terminal = 1.0 - terminateds[step] 
            delta = rewards[step] + self.gamma * next_values[step] * next_non_terminal - values[step]

            next_non_done = 1.0 - np.logical_or(terminateds[step], truncateds[step])
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_done * last_gae_lam

            advantages[step] = last_gae_lam
            
        self.advantages = advantages
        self.returns = advantages + values

    def get_minibatches(self, batch_size: int) -> Generator[GAEExperience, None, None]:
        return self.get_structured_minibatches(batch_size=batch_size)

    def get_structured_minibatches(
        self,
        batch_size: int,
        num_minibatches: int | None = None,
    ) -> Generator[GAEExperience, None, None]:
        """Yields minibatches using (T, B) to (num_minibatches, unroll_length, batch_size, ...).

        This follows the same batching style as the default FPO implementation.
        """
        if len(self.obs) == 0:
            return

        rewards = np.array(self.rewards)
        is_vec = np.asarray(self.rewards[0]).ndim > 0

        if not is_vec:
            rewards = rewards[:, None]

        T, B = rewards.shape[:2]

        obs = np.array(self.obs)
        actions = np.array(self.actions)
        dones = np.array(self.dones)
        values = np.array(self.values)
        returns = np.asarray(self.returns)
        advantages = np.asarray(self.advantages)
        truncateds = np.array(self.truncated)

        if not is_vec:
            obs = obs[:, None, ...]
            actions = actions[:, None, ...]
            dones = dones[:, None]
            values = values[:, None]
            returns = returns[:, None]
            truncateds = truncateds[:, None]
            advantages = advantages[:, None]

        action_infos_stacked = jax.tree_util.tree_map(lambda *x: np.stack(x), *self.action_infos)
        if not is_vec:
            action_infos_stacked = jax.tree_util.tree_map(
                lambda x: np.expand_dims(x, axis=1),
                action_infos_stacked,
            )

        total_transitions = T * B
        if num_minibatches is None:
            if total_transitions < batch_size:
                return
            num_minibatches = total_transitions // batch_size

        if num_minibatches <= 0:
            return

        subseq_count = num_minibatches * batch_size
        if total_transitions % subseq_count != 0:
            raise ValueError(
                f"total transitions {total_transitions} must be divisible by "
                f"num_minibatches*batch_size ({subseq_count})"
            )
        subseq_length = total_transitions // subseq_count
        shuffle_indices = self.rng.permutation(subseq_count)

        def prepare_batch(x: np.ndarray) -> np.ndarray:
            suffix = x.shape[2:]
            x = x.swapaxes(0, 1)  # (T, B, ...) -> (B, T, ...)
            x = x.reshape((-1, subseq_length) + suffix)
            x = x[shuffle_indices, ...]
            x = x.reshape((num_minibatches, batch_size, subseq_length) + suffix)
            x = x.swapaxes(1, 2)
            return x

        obs_mb = prepare_batch(obs)
        actions_mb = prepare_batch(actions)
        rewards_mb = prepare_batch(rewards)
        dones_mb = prepare_batch(dones)
        values_mb = prepare_batch(values)
        returns_mb = prepare_batch(returns)
        advantages_mb = prepare_batch(advantages)
        truncateds_mb = prepare_batch(truncateds)
        action_info_mb = jax.tree_util.tree_map(prepare_batch, action_infos_stacked)

        for i in range(num_minibatches):
            yield GAEExperience(
                obs=obs_mb[i],
                action=actions_mb[i],
                reward=rewards_mb[i],
                done=dones_mb[i],
                next_obs=obs_mb[i],
                value=values_mb[i],
                ret=returns_mb[i],
                adv=advantages_mb[i],
                truncated=truncateds_mb[i],
                action_info=jax.tree_util.tree_map(lambda x: x[i], action_info_mb),
            )

    def clear(self):
        self.reset()
    
    def __len__(self):
        # Total transitions = steps * num_envs
        if not self.obs:
            return 0
        return len(self.obs) * (len(self.obs[0]) if self.obs[0].ndim > 1 else 1)