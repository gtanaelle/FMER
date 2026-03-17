import argparse
from pathlib import Path
import time
from functools import partial
import tomllib

import jax, jax.numpy as jnp
from relax.algorithm.dipo import DIPO
from relax.algorithm.qvpo import QVPO
from relax.algorithm.fmer import FMER
from relax.algorithm.dpmd import DPMD
from relax.algorithm.fpo import FPO
from relax.buffer import TreeBuffer
from relax.network.qvpo import create_qvpo_net
from relax.network.fmer import create_fmer_net
from relax.network.dipo import create_dipo_net
from relax.network.diffv2 import create_diffv2_net
from relax.network.fpo import create_fpo_net
from relax.trainer.off_policy import OffPolicyTrainer
from relax.trainer.on_policy import OnPolicyTrainer
from relax.env import create_env, create_vector_env
from relax.utils.experience import Experience, ObsActionPair, GAEExperience
from relax.utils.fs import PROJECT_ROOT
from relax.utils.random_utils import seeding

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", type=str,default="fmer", help="Algorithm name (dacer, sac, dsact, etc.)")
    parser.add_argument("--env", type=str, default="humanoid", help="Environment name (Hopper, Humanoid, etc.)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    cli_args = parser.parse_args()

    # Construct config path: configs/{alg}/{env}.toml (case-insensitive env name)
    env_name = cli_args.env.lower() if cli_args.env.lower() != cli_args.env.lower().capitalize() else cli_args.env.capitalize()
    config_path = PROJECT_ROOT / "configs" / cli_args.alg.lower() / f"{env_name}.toml"
    
    # Load config
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}\nExpected format: configs/{{alg}}/{{env}}.toml")
    
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)
    
    print(f"Loaded config from: {config_path}")
    
    # Extract all parameters from config
    env_cfg = config.get("environment", {})
    train_cfg = config.get("training", {})
    net_cfg = config.get("network", {})
    alg_cfg = config.get(cli_args.alg.lower(), {})
    trainer_cfg = config.get("trainer", {})
    logging_cfg = config.get("logging", {})
    
    # Create args object with all config values
    class Args:
        pass
    args = Args()
    
    # Set all parameters from config
    args.alg = cli_args.alg.lower()
    args.seed = cli_args.seed
    args.tau = net_cfg.get("tau", 0.1)
    args.num_particles = alg_cfg.get("num_particles", 8)
    args.env = env_cfg.get("env", cli_args.env)
    args.num_vec_envs = env_cfg.get("num_vec_envs", 20)
    args.hidden_num = net_cfg.get("hidden_num", 3)
    args.hidden_dim = net_cfg.get("hidden_dim", 256)
    args.diffusion_steps = net_cfg.get("diffusion_steps", 20)
    args.diffusion_hidden_dim = net_cfg.get("diffusion_hidden_dim", 256)
    args.start_step = train_cfg.get("start_step", int(2e5))
    args.total_step = train_cfg.get("total_step", int(3e7))
    args.lr = train_cfg.get("lr", 1e-4)
    args.batch_size = train_cfg.get("batch_size", 2048)

    
    print(f"Config: alg={args.alg}, env={args.env}, seed={args.seed}")

    master_seed = args.seed
    master_rng, _ = seeding(master_seed)
    env_seed, env_action_seed, eval_env_seed, buffer_seed, init_network_seed, train_seed = map(
        int, master_rng.integers(0, 2**32 - 1, 6)
    )
    init_network_key = jax.random.key(init_network_seed)
    train_key = jax.random.key(train_seed)
    del init_network_seed, train_seed

    # Build env_kwargs for environments that need special parameters
    env_kwargs = {}
    if "env_tasks" in env_cfg:
        env_kwargs["tasks_to_complete"] = env_cfg["env_tasks"]
    
    if args.num_vec_envs > 0:
        env, obs_dim, act_dim = create_vector_env(args.env, args.num_vec_envs, env_seed, env_action_seed, mode="futex", env_kwargs=env_kwargs)
    else:
        env, obs_dim, act_dim = create_env(args.env, env_seed, env_action_seed, env_kwargs=env_kwargs)
    eval_env = None

    hidden_sizes = [args.hidden_dim] * args.hidden_num
    diffusion_hidden_sizes = [args.diffusion_hidden_dim] * args.hidden_num

    buffer = TreeBuffer.from_experience(obs_dim, act_dim, size=int(1e6), seed=buffer_seed)

    gelu = partial(jax.nn.gelu, approximate=False)

    if args.alg == "dipo":
        diffusion_buffer = TreeBuffer.from_example(
            ObsActionPair.create_example(obs_dim, act_dim),
            int(args.total_step),
            int(master_rng.integers(0, 2**32 - 1)),
            remove_batch_dim=False
        )
        TreeBuffer.connect(buffer, diffusion_buffer, lambda exp: ObsActionPair(exp.obs, exp.action))

        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))

        agent, params = create_dipo_net(init_network_key, obs_dim, act_dim, hidden_sizes, num_timesteps=100)
        algorithm = DIPO(agent, params, diffusion_buffer, lr=args.lr, action_gradient_steps=30, policy_target_delay=2, action_grad_norm=0.16)
    elif args.alg == "qvpo":
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        agent, params = create_qvpo_net(init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish,
                                          num_timesteps=args.diffusion_steps,
                                          num_particles=32,
                                          noise_scale=0.1)
        algorithm = QVPO(agent, params, lr=args.lr, alpha_lr=7e-3, delay_alpha_update=250)
    elif args.alg == "fpo":
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        agent, params = create_fpo_net(init_network_key, obs_dim, act_dim,
                                      flow_steps=args.diffusion_steps)
        algorithm = FPO(agent, params)
    elif args.alg == "fmer":
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        agent, params = create_fmer_net(init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish,
                                          num_timesteps=args.diffusion_steps,
                                          num_particles=args.num_particles,
                                          tau=args.tau)
        algorithm = FMER(agent, params, lr=args.lr)
    elif args.alg == 'dpmd':
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        agent, params = create_diffv2_net(init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish,
                                          num_timesteps=args.diffusion_steps, 
                                          num_particles=32, 
                                          noise_scale=0.1,
                                          beta_schedule_scale=0.8)
        algorithm = DPMD(agent, params, lr=args.lr, alpha_lr=7e-3, delay_alpha_update=250, lr_schedule_end=3e-5)
    else:
        raise ValueError(f"Invalid algorithm {args.alg}!")

    
    OUTPUT_DIR = "exp_logs/test"
    if "env_tasks" in env_cfg:
        run_name = args.alg + '_' + f"{args.env}-{env_cfg['nb_task']}" + '_' + time.strftime("%Y%m%d_%H%M%S")  +f'_s{args.seed}'
        log_path = PROJECT_ROOT / "exp_logs/test" / f"{args.env}-{env_cfg['nb_task']}" / run_name
    else:
        run_name = args.alg + '_' + args.env + '_' + time.strftime("%Y%m%d_%H%M%S") +f'_s{args.seed}'
        log_path = PROJECT_ROOT / "exp_logs/test" / args.env / run_name

    use_wandb = logging_cfg.get("use_wandb", False)
    wandb_project = logging_cfg.get("wandb_project", None)
    wandb_config = {
        "alg": args.alg,
        "env": args.env,
        "seed": args.seed,
        "hidden_dim": args.hidden_dim,
        "hidden_num": args.hidden_num,
        "diffusion_steps": args.diffusion_steps,
        "num_vec_envs": args.num_vec_envs,
        "lr": args.lr,
        "tau": args.tau,
    }

    if args.alg == "fpo":
        from relax.buffer.rollout import RolloutBuffer 
        rollout_buffer = RolloutBuffer(gamma=0.99, gae_lambda=0.95, seed=buffer_seed)
        trainer = OnPolicyTrainer(
            env=env,
            algorithm=algorithm,
            buffer=rollout_buffer,             
            mini_batch_size=64,   
            rollout_length=2048,               
            update_epochs=8,      #smaller value for tiny, single-environment buffer to not overfit the flow model to high-variance advantage estimates.            
            total_step=args.total_step,
            log_path=log_path,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_run_name=run_name,
            wandb_config=wandb_config,   
        )
    else:
        trainer = OffPolicyTrainer(
            env=env,
            algorithm=algorithm,
            buffer=buffer,
            start_step=args.start_step,
            total_step=args.total_step,
            sample_per_iteration=1,
            evaluate_env=eval_env,
            warmup_with="random",
            log_path=log_path,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_run_name=run_name,
            wandb_config=wandb_config,
            env_tasks=env_cfg.get("env_tasks", None),
        )
        trainer.setup(Experience.create_example(obs_dim, act_dim, trainer.batch_size))

    trainer.run(train_key)
