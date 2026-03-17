import os
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json

import gymnasium
import gymnasium_robotics
from gymnasium.spaces import Dict
from gymnasium.wrappers import FilterObservation, FlattenObservation
import numpy as np

import setproctitle
from relax.prctl import set_client_pdeathsig
from relax.futex import futex_client_wait, futex_client_notify

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"^gymnasium\.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--index", type=str, required=True)
    parser.add_argument("--seed", type=str, required=True)
    parser.add_argument("--descr", type=str, required=True)
    return parser.parse_args()

def initialize_shm(descr, mode):
    arr = np.memmap(f"/dev/shm/{descr['name']}", dtype=descr["dtype"], mode=mode, shape=tuple(descr["shape"]))
    return arr

def main():
    args = parse_args()

    indices = [int(i) for i in args.index.split(",")]
    seeds = [int(i) for i in args.seed.split(",")]
    assert len(indices) == len(seeds)

    descr = json.loads(args.descr)
    env_kwargs = json.loads(descr.get("env_kwargs", "{}"))

    envs = []
    for seed in seeds:
        env = gymnasium.make(args.env, **env_kwargs)

        # ---------------- 修改区域开始 ----------------
        # 拦截 Dict 空间，保留 observation 键并将其展平为一维 Box，
        # 确保写入共享内存的数据是标准的一维 numpy 数组
        if isinstance(env.observation_space, Dict):
            env = FlattenObservation(FilterObservation(env, ['observation']))
        # ---------------- 修改区域结束 ----------------

        env.reset(seed=seed)
        envs.append(env)

    setproctitle.setproctitle(f"FutexWorker:{indices[0]}")
    set_client_pdeathsig()

    action = initialize_shm(descr["action"], "r")
    obs = initialize_shm(descr["obs"], "r+")
    obs2 = initialize_shm(descr["obs2"], "r+")
    reward = initialize_shm(descr["reward"], "r+")
    terminated = initialize_shm(descr["terminated"], "r+")
    truncated = initialize_shm(descr["truncated"], "r+")
    signal = initialize_shm(descr["signal"], "r+")
    signal_pointer = signal.ctypes.data

    for i, env in zip(indices, envs):
        obs[i], _ = env.reset()
    futex_client_notify(signal_pointer)

    last = 0

    try:
        while True:
            last = futex_client_wait(signal_pointer, last)
            command = last & 0b11
            if command == 0b01:
                for i, env in zip(indices, envs):
                    _obs2, _reward, _terminated, _truncated, _ = env.step(action[i])
                    obs2[i] = _obs2
                    reward[i] = _reward
                    terminated[i] = _terminated
                    truncated[i] = _truncated
                    if _terminated or _truncated:
                        obs[i], _ = env.reset()
                    else:
                        obs[i] = _obs2
                futex_client_notify(signal_pointer)
            elif command == 0b10:
                for env in envs:
                    env.close()
                break
            else:
                raise ValueError(f"Unknown command: {command}")
    except KeyboardInterrupt:
        pass
    finally:
        futex_client_notify(signal_pointer)

if __name__ == "__main__":
    main()
