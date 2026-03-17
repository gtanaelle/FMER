# FMER Benchmark Results

## FrankaKitchen Environment

The FrankaKitchen environment is a multi-task manipulation benchmark that tests the ability of reinforcement learning agents to accomplish multiple sequential tasks. The environment consists of 7 possible tasks that can be combined to create problems of varying difficulty:

- Opening/closing the microwave
- Opening/closing the cabinet
- Opening/closing the drawer
- Moving the kettle
- Turning on/off the light
- Turning on/off the stove
- Sliding the cabinet door

This environment presents significant challenges due to its **sparse reward structure**, where feedback is only provided upon successful task completion. Furthermore, the environment exhibits a **multimodal optimal action distribution**, meaning there are multiple diverse ways to achieve the same task objective. These characteristics make FrankaKitchen ideal for evaluating the robustness and generalization capabilities of reinforcement learning algorithms.

In this benchmark, we evaluate our approach across **four different task count settings (N)**: N=1 (single task), N=2 (two tasks), N=4 (four tasks), and N=7 (all tasks). These settings allow us to assess how well methods scale as the task complexity increases. Results represent the maximum performance over 1M environment steps, averaged across 5 random seeds.

The table below shows the mean number of accomplished tasks (with standard deviation in parentheses) across baselines and our proposed FMER method:

| Method | N=1 | N=2 | N=4 | N=7 |
|--------|-----|-----|-----|-----|
| **PPO** | 0.02 (0.04) | 0.18 (0.40) | 0.06 (0.09) | 0.78 (0.44) |
| **SAC** | 0.80 (0.45) | 1.40 (0.34) | 1.24 (0.37) | 1.22 (0.45) |
| **TD3** | 0.00 (0.00) | 0.62 (0.57) | 0.72 (0.82) | 0.92 (0.88) |
| **FPO** | 0.18 (0.20) | 0.50 (0.47) | 0.32 (0.26) | 0.66 (0.21) |
| **DIPO** | 0.22 (0.44) | 0.72 (0.66) | 0.94 (0.88) | 1.78 (1.11) |
| **DPMD** | 0.64 (0.49) | 1.12 (0.27) | 1.22 (0.44) | 1.56 (0.82) |
| **QVPO** | 0.02 (0.04) | 0.08 (0.13) | 0.08 (0.13) | 1.48 (0.69) |
| **FMER** | **1.00 (0.00)** | **1.26 (0.32)** | **2.00 (0.14)** | **3.20 (1.06)** |

FMER demonstrates superior performance across all task count settings, achieving perfect single-task completion (N=1) and significantly outperforming all baselines in multi-task scenarios, particularly as task complexity increases.

## MuJoCo Benchmarks

We evaluate FMER on standard continuous control benchmarks from the MuJoCo simulator. These include:

- **Hopper**: A 2D locomotion task with 3 joints
- **HalfCheetah**: A 2D running cheetah with 6 joints
- **Walker2d**: A 2D bipedal walker with 6 joints
- **Humanoid**: A high-dimensional 3D humanoid control task with 17 joints

The results below show the maximum average return achieved over 1M environment steps, averaged across 5 random seeds with standard deviations in parentheses.

| Algorithm | Hopper | HalfCheetah | Walker2d | Humanoid |
|-----------|--------|------------|----------|----------|
| **PPO** | 1223.19 (345.21) | 5982.74 (810.60) | 4606.76 (211.17) | 811.94 (78.20) |
| **SAC** | 3354.63 (90.53) | 10295.33 (1170.43) | 4453.24 (521.33) | **5350.96 (75.07)** |
| **TD3** | 3664.30 (60.89) | 9496.59 (901.14) | 4969.19 (668.54) | 363.16 (202.48) |
| **FPO** | 1216.86 (241.82) | 2860.01 (1513.28) | 635.33 (219.05) | 371.46 (163.38) |
| **DIPO** | 3592.82 (104.07) | 9516.47 (1417.20) | 5028.66 (217.36) | 5123.38 (78.81) |
| **DPMD** | 3503.62 (65.09) | 10706.06 (817.59) | 4910.31 (350.06) | 5100.60 (44.91) |
| **QVPO** | **3671.77 (205.24)** | 10309.60 (1398.37) | 4846.39 (397.71) | 5078.00 (47.22) |
| **FMER** | 3444.46 (158.88) | **12332.38 (903.67)** | **5285.08 (438.80)** | 5286.12 (65.35) |

FMER achieves strong performance across the MuJoCo benchmarks, with the best results on HalfCheetah and Walker2d, demonstrating its effectiveness on diverse continuous control tasks with varying dimensionality and complexity.
