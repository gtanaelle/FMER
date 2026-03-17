from pathlib import Path
import subprocess
import sys
import os
from typing import Callable, Optional, Tuple

import jax
import numpy as np
from gymnasium import Env
from tqdm import tqdm
from tensorboardX import SummaryWriter
from tensorboardX.summary import hparams

from relax.algorithm import Algorithm
from relax.buffer import ExperienceBuffer # Ensure this is your RolloutBuffer implementation
from relax.env.vector import VectorEnv
from relax.trainer.accumulator import SampleLog, VectorSampleLog, UpdateLog, Interval
from relax.utils.experience import GAEExperience


class OnPolicyTrainer:
    def __init__(
        self,
        env: Env,
        algorithm: Algorithm,
        buffer: ExperienceBuffer,
        log_path: Path,
        rollout_length: int = 2048,
        update_epochs: int = 10,
        mini_batch_size: int = 64,
        total_step: int = int(1e6),
        evaluate_env: Optional[Env] = None,
        evaluate_every: int = 10000,
        evaluate_n_episode: int = 10,
        sample_log_n_episode: int = 10,
        update_log_n_step: int = 1, 
        done_info_keys: Tuple[str, ...] = (),
        hparams: Optional[dict] = None,
        policy_pkl_template: str = "policy-{sample_step}-{update_step}.pkl",
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_config: Optional[dict] = None,
    ):
        self.env = env
        self.algorithm = algorithm
        self.buffer = buffer
        
        self.rollout_length = rollout_length
        self.update_epochs = update_epochs
        self.mini_batch_size = mini_batch_size
        self.total_step = total_step
        
        self.log_path = log_path
        self.policy_pkl_template = policy_pkl_template
        self.evaluate_env = evaluate_env
        self.evaluate_every = evaluate_every
        self.evaluate_n_episode = evaluate_n_episode
        self.sample_log_n_episode = sample_log_n_episode
        self.update_log_n_step = update_log_n_step
        self.done_info_keys = done_info_keys
        self.hparams = hparams
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.wandb_config = wandb_config
        self._wandb = None

        if isinstance(self.env.unwrapped, VectorEnv):
            self.is_vec = True
            self.sample_log = VectorSampleLog(self.env.unwrapped.num_envs)
        else:
            self.is_vec = False
            self.sample_log = SampleLog()
            
        self.update_log = UpdateLog()
        self.last_metrics = {}
        
        self.sample_log_interval = Interval(self.sample_log_n_episode)
        self.evaluate_interval = Interval(self.evaluate_every)
        self.best_eval_reward = -float("inf")

    def setup(self, dummy_obs: np.ndarray):
        self.progress = tqdm(total=self.total_step, desc="Sample Step", disable=None, dynamic_ncols=True)

        if not self.log_path.exists():
            self.log_path.mkdir(parents=True, exist_ok=True)
            
        self.algorithm.save_policy_structure(self.log_path, dummy_obs)
        
        # Start the background evaluator subprocess
        self.evaluator = subprocess.Popen(
            [
                sys.executable,
                "-m", "relax.trainer.evaluator",
                str(self.log_path),
                "--env", self.env.spec.id,
                "--num_episodes", str(self.evaluate_n_episode),
                "--seed", str(0),
            ],
            stdin=subprocess.PIPE,
            bufsize=0,
        )

        if self.use_wandb:
            try:
                import wandb
                self._wandb = wandb
                wandb.init(project=self.wandb_project, name=self.wandb_run_name, config=self.wandb_config)
            except Exception as e:
                print(f"[wandb] init failed: {e}")
                self.use_wandb = False

    def collect_rollout(self, key: jax.Array, obs: np.ndarray) -> np.ndarray:
        sl = self.sample_log
        step_key_fn = jax.jit(lambda k, s: jax.random.fold_in(k, s))

        for step in range(self.rollout_length):
            step_key = step_key_fn(key, step)
            
            action, value, action_info = self.algorithm.step(step_key, obs)
            env_action = (
                np.tanh(action)
                if getattr(self.algorithm, "squash_action_before_step", False)
                else action
            )
            next_obs, reward, terminated, truncated, info = self.env.step(env_action)

            self.buffer.add(obs, action, reward, value, terminated, truncated, action_info)

            any_done = sl.add(reward, terminated, truncated, info)

            if any_done and self.sample_log_interval.check(sl.sample_episode):
                sl.log(self.add_scalar)

            # VectorEnv handles auto-reset automatically via Gymnasium
            if self.is_vec:
                obs = next_obs
            else:
                if any_done:
                    obs, _ = self.env.reset()
                else:
                    obs = next_obs
            
            # Smoothly update progress bar within the loop
            self.progress.update(self.env.num_envs if self.is_vec else 1)
    
        # 1. Get value of the current obs (might be a reset obs for some envs)
        next_value = self.algorithm.get_value(obs)
        
        # 2. Correct bootstrapping for truncated environments in VectorEnv
        if self.is_vec and "final_observation" in info:
            # Look at the last step stored in buffer to see which were truncated
            last_truncated = self.buffer.truncated[-1] 
            final_mask = info.get("_final_observation", np.zeros(self.env.num_envs, dtype=bool))
            
            for i in range(self.env.num_envs):
                # If a sub-env ended and it was a truncation, we need the true final value
                if final_mask[i] and last_truncated[i]:
                    real_next_obs = info["final_observation"][i]
                    real_next_val = self.algorithm.get_value(np.expand_dims(real_next_obs, 0))[0]
                    next_value[i] = real_next_val

        # 3. Compute GAE with corrected next_value
        self.buffer.compute_returns_and_advantages(next_value)
        return obs

    def update_policy(self, update_key: jax.Array):
        ul = self.update_log
        epoch_key_fn = jax.jit(lambda k, e: jax.random.fold_in(k, e))

        total_samples = len(self.buffer)
        num_minibatches = total_samples // self.mini_batch_size

        if num_minibatches == 0:
            return

        if hasattr(self.algorithm, "update_obs_stats"):
            self.algorithm.update_obs_stats(np.array(self.buffer.obs))

        for epoch in range(self.update_epochs):
            epoch_key = epoch_key_fn(update_key, epoch)
            batch_keys = jax.random.split(epoch_key, num_minibatches)

            for i, batch in enumerate(
                self.buffer.get_structured_minibatches(
                    batch_size=self.mini_batch_size,
                    num_minibatches=num_minibatches,
                )
            ):
                info = self.algorithm.update(batch_keys[i], batch)
                ul.add(info)

        if ul.update_step % self.update_log_n_step == 0:
            ul.log(self.add_scalar)
            
        # Clear buffer to maintain on-policy requirements
        self.buffer.clear()

    def train(self, key: jax.Array):
        obs, _ = self.env.reset()
        # Initialize with a single sample's shape
        self.setup(obs[0] if self.is_vec else obs)

        iter_key_fn = create_iter_key_fn(key)
        sl, ul = self.sample_log, self.update_log

        self.progress.unpause()
        
        iteration = 0
        while sl.sample_step <= self.total_step:
            iteration += 1
            rollout_key, update_key = iter_key_fn(iteration)

            obs = self.collect_rollout(rollout_key, obs)
            self.update_policy(update_key)

            # Evaluate at intervals or on the very last step
            if self.evaluate_interval.check(sl.sample_step) or sl.sample_step >= self.total_step:
                policy_pkl_name = self.policy_pkl_template.format(
                    sample_step=sl.sample_step,
                    update_step=ul.update_step,
                )
                self.algorithm.save_policy(self.log_path / policy_pkl_name)

                command = f"{sl.sample_step},{self.log_path / policy_pkl_name}\n"
                self.evaluator.stdin.write(command.encode())

                if os.path.exists(self.log_path / "evaluation.csv"):
                    try:
                        with open(self.log_path / "evaluation.csv", "r") as f:
                            lines = f.readlines()
                            if lines:
                                last_line = lines[-1]
                                items = last_line.strip().split(",")
                                step, ep_len, ep_ret = int(items[0]), float(items[1]), float(items[2])
                                if ep_ret > self.best_eval_reward:
                                    self.best_eval_reward = ep_ret
                                if self.use_wandb and self._wandb is not None:
                                    self._wandb.log({"evaluate/episode_length": ep_len}, step=sl.sample_step)
                                    self._wandb.log({"evaluate/episode_reward": ep_ret}, step=sl.sample_step)
                                    self._wandb.log({"evaluate/best_reward": self.best_eval_reward}, step=sl.sample_step)
                    except Exception:
                        pass # Catch potential IO race conditions from the evaluator

    def add_scalar(self, tag: str, value: float, step: int):
        self.last_metrics[tag] = value
        if self.use_wandb and self._wandb is not None:
            wandb_step = getattr(self.sample_log, "sample_step", step)
            self._wandb.log({tag: value}, step=wandb_step)

    def run(self, key: jax.Array):
        try:
            self.train(key)
        except KeyboardInterrupt:
            pass
        finally:
            self.finish()

    def finish(self):
        self.env.close()
        self.algorithm.save(self.log_path / "state.pkl")
        if self.hparams is not None and len(self.last_metrics) > 0:
            exp, ssi, sei = hparams(self.hparams, self.last_metrics)
        self.progress.close()
        
        # Clean shutdown of evaluation subprocess
        if hasattr(self, 'evaluator'):
            try:
                self.evaluator.stdin.close()
                self.evaluator.wait(timeout=5)
            except Exception:
                self.evaluator.kill()
            
        if self.use_wandb and self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass


def create_iter_key_fn(key: jax.Array) -> Callable[[int], Tuple[jax.Array, jax.Array]]:
    def iter_key_fn(step: int):
        iter_key = jax.random.fold_in(key, step)
        sample_key, update_key = jax.random.split(iter_key)
        return sample_key, update_key

    iter_key_fn = jax.jit(iter_key_fn)
    iter_key_fn(0)  # Warm up
    return iter_key_fn