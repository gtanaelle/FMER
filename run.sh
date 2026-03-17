#!/bin/sh
#
#SBATCH --job-name=diff_rl
#SBATCH --partition=gpu
#SBATCH --output=slurm_jobs/diff_rl.log
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=5G
#SBATCH --account=research-ceg-tp

# Redirect output to timestamped log file
LOGFILE="slurm_jobs/$(date +%Y%m%d_%H%M)_${SLURM_JOB_ID}.log"
exec > "$LOGFILE" 2>&1

# 1. 物理路径
PROJECT_ROOT="/scratch/tinggao/GVPO"
cd $PROJECT_ROOT

# 2. 清理并加载模块 (注意：如果你已经有很好的驱动，module load 可能不是必须的，但建议保留)
module purge
module load 2025
module load cuda/12.1 

# Set compiler
export CC=gcc
export CXX=g++

# Optional: check compiler
which gcc
echo "Using compiler: $CC"

# 3. 设置 PATH：确保原有的 $PATH 放在最后，且不破坏系统路径
export PATH="$HOME/.local/bin:$PROJECT_ROOT/.venv/bin:$PATH"
export XLA_FLAGS=--xla_gpu_deterministic_ops=true
export TF_CUDNN_DETERMINISTIC=1

# 4. 关键：修复 LD_LIBRARY_PATH (增加 nvjitlink)
VENV_LIB="$PROJECT_ROOT/.venv/lib/python3.13/site-packages/nvidia"
export LD_LIBRARY_PATH="$VENV_LIB/cusparse/lib:$VENV_LIB/nvjitlink/lib:$VENV_LIB/cudnn/lib:$VENV_LIB/cublas/lib:$VENV_LIB/cuda_runtime/lib:$LD_LIBRARY_PATH"

# 5. 调试：不再使用 srun，直接用 python 检查
echo "--- Testing JAX with direct python call ---"
uv run python -c "import jax; print('JAX Devices:', jax.devices())"

# 6. 运行训练：去掉 srun
echo "--- Starting Training ---"

# Replace SWEEP_ID with your actual sweep ID from: uv run wandb sweep weep_config.yaml
# SWEEP_ID="MyDiffusionRL/GVPO/7wbghw4h"
SWEEP_ID="MyDiffusionRL/enhance-test/3sbfttt7"
# SWEEP_ID="MyDiffusionRL/baseline/sikwlhvi"
uv run wandb agent $SWEEP_ID


# uv run python scripts/train_mujoco.py --alg gvpo --env humanoid --noise_scale 0.0 --seed 0

# # source ./.venv/bin/activate
# # srun uv run python -c "import jax; print('Available devices:', jax.devices())"
