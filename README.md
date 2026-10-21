<div align="center">

# FMER

### Flow Matching Policy Optimization with Mirror Descent and Entropy Constraints

**Ting Gao, Stavros Orfanoudakis, Nan Lin, Winnie Daamen, Serge Hoogendoorn, Elvin Isufi**<br>
Delft University of Technology

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-4b44ce.svg)](https://neurips.cc/Conferences/2026)
[![Blog](https://img.shields.io/badge/blog-post-ff6f61.svg)](https://gtanaelle.github.io/blog/2026/10/20/flow-matching-policy-optimization/)
[![Python 3.13](https://img.shields.io/badge/python-3.13-3776ab.svg?logo=python&logoColor=white)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/built%20with-JAX-a259ff.svg)](https://github.com/jax-ml/jax)

<img src="assets/kitchen_fmer_vs_dipo.gif" width="90%" alt="FMER vs DIPO rollouts on FrankaKitchen"/>

<sub>FrankaKitchen, 280-step budget. <b>FMER</b> finishes 5 sub-tasks (kettle → bottom burner → light switch → top burner → slide cabinet); the strongest baseline, <b>DIPO</b>, finishes 3.</sub>

</div>

---

## 💡 TL;DR

FMER is an online RL algorithm with a **flow matching (ODE) policy** that gets three properties no prior diffusion or flow policy has all at once:

| | Policy Mirror Descent | Entropy Constrained | Unbiased Entropy Objective | Simulation-Free Training |
|:--|:--:|:--:|:--:|:--:|
| DIPO | ✗ | ✗ | ✗ | ✓ |
| DPMD | ✓ | ✗ | ✗ | ✓ |
| QVPO | ✗ | ✓ | ✗ | ✓ |
| SAC-Flow | ✗ | ✓ | ✗ | ✗ |
| FLAME | ✗ | ✓ | ✗ | ✗ |
| **FMER (ours)** | ✓ | ✓ | ✓ | ✓ |

On the sparse-reward **FrankaKitchen** benchmark FMER outperforms eight baselines in nearly every setting, and the gap widens as the number of tasks grows. It stays competitive on dense-reward **MuJoCo**.

## 👩🏻‍🔬 Method

FMER represents the policy as a velocity field $v_\theta(x_t, t, s)$ in an unbounded latent space and squashes the result into bounded actions with $a = \tanh(x_1)$. Training minimizes one loss:

```math
\mathcal{L}_{\text{FMER}}(\theta) = {\color{#1f77b4}\mathcal{L}_{\text{W-CFM}}(\theta)} + \alpha\,{\color{#2ca02c}\mathcal{L}_{\text{ent}}(\theta)}
```

**1. Mirror descent through weighted flow matching (Thm. 1).** Sample $M$ candidate actions from the current policy, score them with the critic, and weight the conditional flow matching loss by the softmax of their advantages:

```math
{\color{#1f77b4}\mathcal{L}_{\text{W-CFM}} = \mathbb{E}\left[ w(s,a)\,\big\| v_\theta(x_t,t,s) - (\text{arctanh}(a) - x_0) \big\|^2 \right]},
\qquad w(s,a^{(j)}) = \frac{\exp(A(s,a^{(j)})/\tau)}{\sum_l \exp(A(s,a^{(l)})/\tau)}
```

We prove that the global minimizer is exactly the policy mirror descent target $\pi^\ast \propto \pi_k \exp(A/\tau)$. Training is simulation-free, so gradients never pass through the ODE solver.

**2. Closed-form entropy with a tanh correction (Thm. 2, Prop. 1).** The entropy of an ODE policy follows from the divergence of its vector field. We correct for the density distortion that the tanh squashing introduces:

```math
{\color{#2ca02c}\mathcal{L}_{\text{ent}} = -\,\mathbb{E}_{t,x_t}\Big[ \nabla_x \cdot v_\theta(x_t,t,s) \;-\; 2\,\langle \tanh(x_t),\, v_\theta(x_t,t,s) \rangle \Big]}
```

The divergence is estimated with Hutchinson's trace estimator, which costs $\mathcal{O}(d)$. The multiplier $\alpha$ is tuned by dual gradient descent toward a target entropy $\bar H$.

**3. Adaptive trust region via effective sample size.** The temperature $\tau$ is adjusted at every step to hold the effective sample size $\mathrm{ESS} = 1/\sum_l w_l^2$ of the candidate weights at a target $\mathrm{ESS}^\ast$. This removes per-environment temperature tuning.

During interaction, FMER samples $M$ candidates with Euler integration (10 steps) and executes the one with the highest Q-value.

## 📊 Results

FMER against the **strongest of eight baselines** (PPO, SAC, TD3, DIPO, DPMD, QVPO, SAC-Flow, FLAME) in each setting. Each number is a mean over 5 seeds after 1M environment steps; full tables with standard deviations are in the paper.

**FrankaKitchen**: sparse reward, task completion rate (%) with $N$ sub-tasks per goal

| | N=1 | N=2 | N=4 | N=7 |
|:--|:--:|:--:|:--:|:--:|
| Best baseline | 60.0 <sub>DPMD</sub> | **50.0** <sub>DPMD</sub> | 25.0 <sub>DPMD</sub> | 23.1 <sub>DIPO</sub> |
| **FMER** | **80.0** | 48.0 | **46.5** | **40.3** |

**MuJoCo-v5**: dense reward, episode return

| | Hopper | HalfCheetah | Walker2d | Humanoid |
|:--|:--:|:--:|:--:|:--:|
| Best baseline | **3662** <sub>TD3</sub> | 10901 <sub>FLAME</sub> | 5234 <sub>SAC-Flow</sub> | **5327** <sub>SAC</sub> |
| **FMER** | 2913 | **12332** | **5283** | 5286 |

On FrankaKitchen, FMER's lead grows with the number of sub-tasks: at $N=7$ it completes almost twice as many as the best baseline. On MuJoCo it is competitive and wins on HalfCheetah and Walker2d. Training on Humanoid takes 215 min on one V100, faster than the diffusion baselines (276–294 min).

## 🚀 Getting started

### Installation

Requirements: Linux with an NVIDIA GPU (CUDA 12). The pinned `jax[cuda12]` wheels are Linux-only. The project uses [uv](https://docs.astral.sh/uv/) and Python 3.13.

```bash
git clone https://github.com/gtanaelle/FMER.git
cd FMER
uv sync
```

### Training

Each run reads its configuration from `configs/<alg>/<env>.toml`:

```bash
# FMER on Walker2d, seed 0
uv run python scripts/train_mujoco.py --alg fmer --env walker2d --seed 0

# FMER on FrankaKitchen (all 7 tasks); use frankakitchen_1 / _2 / _4 for N = 1, 2, 4
uv run python scripts/train_mujoco.py --alg fmer --env frankakitchen --seed 0
```

Available environments: `hopper`, `halfcheetah`, `walker2d`, `humanoid`, `frankakitchen`, `frankakitchen_1`, `frankakitchen_2`, `frankakitchen_4`.
The baselines included in this repo (`dipo`, `dpmd`, `qvpo`, `fpo`) run with the same command. Not every baseline has a config for every environment; for `dpmd` and `qvpo`, Walker2d is `--env walker`.

Logs, checkpoints and `evaluation.csv` (evaluated every 10K steps) are written to `exp_logs/test/<env>/<run_name>/`.

All configs ship with `use_wandb = true`, so run `uv run wandb login` first, or set `use_wandb = false` in the `[logging]` section of the config to train without W&B.

### Sweeps with Weights & Biases

```bash
uv run wandb sweep weep_config.yaml     # creates a sweep over envs x seeds
uv run wandb agent <entity/project/sweep_id>
```

The sweep grid (environments × seeds 0–4) is defined in `weep_config.yaml`. Each run logs to the `wandb_project` set in its config.

## 📁 Repository structure

```
configs/        per-algorithm, per-environment TOML configs
relax/
  algorithm/    FMER and baseline algorithms (fmer.py, dipo.py, dpmd.py, qvpo.py, fpo.py)
  network/      policy and critic networks
  utils/FM.py   flow matching utilities
  trainer/      off-/on-policy trainers and evaluator
  env/ buffer/  vectorized environments and replay buffers
scripts/        training entry point, result collection and plotting
```

## 📝 Citation

```bibtex
@inproceedings{fmer2026,
  title     = {Flow Matching Policy Optimization with Mirror Descent and Entropy Constraints},
  author    = {Gao, Ting and Orfanoudakis, Stavros and Lin, Nan and Daamen, Winnie and Hoogendoorn, Serge and Isufi, Elvin},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

