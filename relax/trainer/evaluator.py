import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
from pathlib import Path
import argparse
import pickle
import csv

import numpy as np
import jax

from relax.env import create_env
from relax.utils.persistence import PersistFunction

def evaluate(env, policy_fn, policy_params, num_episodes, obs_mean=None, obs_std=None, squash_action=False):
    ep_len_list = []
    ep_ret_list = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        ep_len = 0
        ep_ret = 0.0
        while True:
            if obs_mean is not None and obs_std is not None:
                policy_obs = (obs - obs_mean) / obs_std
            else:
                policy_obs = obs
            act = policy_fn(policy_params, policy_obs)
            if squash_action:
                act = np.tanh(act)
            else:
                act = np.clip(act, -1, 1)
            obs, reward, terminated, truncated, _ = env.step(act)
            ep_len += 1
            ep_ret += reward
            if terminated or truncated:
                break
        ep_len_list.append(ep_len)
        ep_ret_list.append(ep_ret)
    return ep_len_list, ep_ret_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("policy_root", type=Path)
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--env_tasks", type=str, nargs='+', default=None, help="Tasks for FrankaKitchen environment")
    args = parser.parse_args()

    best_reward = -float("inf")
    master_rng = np.random.default_rng(args.seed)
    env_seed, env_action_seed, policy_seed = map(int, master_rng.integers(0, 2**32 - 1, 3))
    
    # Build env_kwargs
    env_kwargs = {}
    if args.env_tasks is not None:
        env_kwargs["tasks_to_complete"] = args.env_tasks
    
    env, _, _ = create_env(args.env, env_seed, env_action_seed, env_kwargs=env_kwargs)

    policy = PersistFunction.load(args.policy_root / "deterministic.pkl")
    @jax.jit
    def policy_fn(policy_params, obs):
        return policy(policy_params, obs) #.clip(-1, 1)

    while payload := sys.stdin.readline():
        step, policy_path = payload.strip().split(",", maxsplit=1)
        # step = int(step) // 20
        with open(policy_path, "rb") as f:
            loaded = pickle.load(f)

        obs_mean = None
        obs_std = None
        is_fpo_run = args.policy_root.name.startswith("fpo_")
        if is_fpo_run and isinstance(loaded, dict) and "policy" in loaded:
            policy_params = loaded["policy"]
            if "obs_mean" in loaded and "obs_std" in loaded:
                obs_mean = np.asarray(loaded["obs_mean"], dtype=np.float32)
                obs_std = np.asarray(loaded["obs_std"], dtype=np.float32)
        else:
            policy_params = loaded

        ep_len_list, ep_ret_list = evaluate(
            env,
            policy_fn,
            policy_params,
            args.num_episodes,
            obs_mean=obs_mean,
            obs_std=obs_std,
            squash_action=is_fpo_run,
        )
        ep_len_mean = np.array(ep_len_list)
        ep_ret_mean = np.array(ep_ret_list)

        if ep_ret_mean.mean() > best_reward:
            best_reward = ep_ret_mean.mean()
            best_policy_path = args.policy_root / "best_policy.pkl"
            with open(best_policy_path, "wb") as f:
                pickle.dump(policy_params, f)

        # Append results to CSV in policy_root for easy inspection.
        try:
            csv_path = args.policy_root / "evaluation.csv"
            write_header = not csv_path.exists()
            with open(csv_path, "a", newline="") as _csvf:
                _writer = csv.writer(_csvf)
                if write_header:
                    _writer.writerow(["step", "length", "return"])
                _writer.writerow([step, float(ep_len_mean.mean()), float(ep_ret_mean.mean())])
        except Exception:
            pass
