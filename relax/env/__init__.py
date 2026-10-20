import numpy as np
import gymnasium_robotics  # 【修改点】确保在顶部导入，以便全局注册 Adroit 等机器人环境
from gymnasium import Env, Wrapper, make
from gymnasium.spaces import Box, Dict # 【修改点】引入 Dict 空间

from relax.env.vector import VectorEnv, SerialVectorEnv, GymProcessVectorEnv, PipeProcessVectorEnv, SpinlockProcessVectorEnv, FutexProcessVectorEnv

class RelaxWrapper(Wrapper):
    def __init__(self, env: Env, action_seed: int = 0):
        # Only call super().__init__ for regular Env, not VectorEnv
        if not isinstance(env, VectorEnv):
            super().__init__(env)
        else:
            self.env = env
            
        # 【修改点 1】兼容 Box (MuJoCo) 和 Dict (Adroit) 观测空间
        if isinstance(env.observation_space, Box):
            self.is_dict_obs = False
            base_obs_space = env.observation_space
        elif isinstance(env.observation_space, Dict):
            self.is_dict_obs = True
            base_obs_space = env.observation_space.spaces['observation']
        else:
            raise TypeError("Observation space must be a Box or Dict containing an 'observation' Box.")

        assert isinstance(env.action_space, Box) and env.action_space.is_bounded()

        # 【修改点 2】基于基础观测空间安全地提取维度
        if isinstance(env, VectorEnv):
            _, self.obs_dim = base_obs_space.shape
            _, self.act_dim = env.action_space.shape
            single_action_space = env.single_action_space
        else:
            self.obs_dim, = base_obs_space.shape
            self.act_dim, = env.action_space.shape
            single_action_space = env.action_space

        if np.any(single_action_space.low != -1.0) or np.any(single_action_space.high != 1.0):
            print(f"NOTE: The action space is not normalized, but {single_action_space.low} to {single_action_space.high}, will be rescaled.")
            self.needs_rescale = True
            self.original_action_center = (single_action_space.low + single_action_space.high) * 0.5
            self.original_action_half_range = (single_action_space.high - single_action_space.low) * 0.5
        else:
            self.needs_rescale = False
            
        self.original_action_dtype = env.action_space.dtype

        # 【修改点 3】显式覆盖 observation_space，让下游 RL 网络只看到一个扁平的 Box
        self.observation_space = Box(
            low=base_obs_space.low,
            high=base_obs_space.high,
            shape=base_obs_space.shape,
            dtype=np.float32
        )
        
        # 【修改点 4】使用 action_space 替代旧版 Gym 的 _action_space
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=env.action_space.shape,
            dtype=np.float32,
            seed=action_seed
        )

    @property
    def spec(self):
        return self.env.spec
        
    def _extract_obs(self, obs):
        """【修改点 5】内部辅助方法：在环境步进时，如果遇到字典就剥离出纯 numpy 数组"""
        if self.is_dict_obs:
            obs = obs['observation']
        return obs.astype(np.float32, copy=False)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._extract_obs(obs), info  # 使用辅助方法处理 obs

    def step(self, action: np.ndarray):
        action = action.astype(self.original_action_dtype)
        if self.needs_rescale:
            action *= self.original_action_half_range
            action += self.original_action_center
            
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._extract_obs(obs), reward, terminated, truncated, info

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)
        
def create_env(name: str, seed: int, action_seed: int = 0, env_kwargs: dict = None):
    import gymnasium_robotics # 确保作用域内注册环境
    if env_kwargs is None:
        env_kwargs = {}
    env = make(name, **env_kwargs)
    env.reset(seed=seed)
    env = RelaxWrapper(env, action_seed)
    return env, env.obs_dim, env.act_dim

def create_vector_env(name: str, num_envs: int, seed: int, action_seed: int = 0, mode: str = "serial", env_kwargs: dict = None, **kwargs):
    import gymnasium_robotics # 确保作用域内注册环境
    
    implementations = {
        "serial": SerialVectorEnv,
        "gym": GymProcessVectorEnv,
        "pipe": PipeProcessVectorEnv,
    }
    if SpinlockProcessVectorEnv is not None:
        implementations["spinlock"] = SpinlockProcessVectorEnv
    if FutexProcessVectorEnv is not None:
        implementations["futex"] = FutexProcessVectorEnv
    
    if mode == "futex" and "futex" not in implementations:
        mode = "serial"  # futex C extension not built; fall back to pure-Python envs
    if mode not in implementations:
        available = ", ".join(implementations.keys())
        raise ValueError(f"Mode '{mode}' not available. Available modes: {available}. "
                        f"Some modes require C extensions to be built (run: pip install -e .).")
    
    if env_kwargs is None:
        env_kwargs = {}
    Impl = implementations[mode]
    env = Impl(name, num_envs, seed, env_kwargs=env_kwargs, **kwargs)
    env = RelaxWrapper(env, action_seed)
    return env, env.obs_dim, env.act_dim