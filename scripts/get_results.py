import argparse
from pathlib import Path
import pandas as pd
import wandb
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def read_evaluation():
    eval_results = []

    ########################################
    # Read results for diffusion-based baselines
    for env_folder in Path(baseline_logs_path).iterdir():
        if not env_folder.is_dir():
            continue
        env = env_folder.name
        
        # Read results stored in baseline_logs_path
        for policy in ["dipo", "dpmd", "fpo"]:
            folders = list(Path(f"{baseline_logs_path}{env}/").glob(f"{policy}*"))
            if len(folders)!=5:
                print(f"{policy} | {env}, found {len(folders)} runs.")
                continue
            for folder in folders:
                df = pd.read_csv(f"{folder}/evaluation.csv")
                eval_results.append({
                    "env": env,
                    "policy": policy,
                    "seed":int(folder.name[-1]),
                    "return": df["return"].max(),
                    # "best_of_last": df[df["step"]>=1e6*0.9]["return"].max()
                })

        # Read results for qvpo
        qvpo_folders = list(Path(f"{qvpo_logs_path}{env}/").glob(f"*"))
        if len(qvpo_folders)!=5:
            print(f"qvpo | {env}, found {len(qvpo_folders)} runs.")
            continue
        for folder in qvpo_folders:
            df = pd.read_csv(f"{folder}/eval_rewards.csv")
            seed = folder.name[-1]
            eval_results.append({
                "env": env,
                "policy": "qvpo",
                "seed":int(seed),
                "return": df["mean_reward"].max(),
                # "best_of_last": df[df["step"]>=1e6*0.9]["mean_reward"].max()
            })
    
    ########################################
    # Read results for FMER
    fmer_folders = list(Path(f"{FMER_logs_path}").glob(f"*/*"))
    for folder in fmer_folders:
        if not folder.is_dir():
            continue
        env = folder.parent.name
        df = pd.read_csv(f"{folder}/evaluation.csv")
        eval_results.append({
            "env": env,
            "policy": "fmer",
            "seed":int(folder.name[-1]),
            "return": df["return"].max(),
        })

    ########################################
    # Read results for traditional baselines
    ppo_td3_df = pd.read_csv(baseline_logs_path+"ppo_td3_eval.csv").rename(columns={"environment":"env", "eval_mean_reward":"return", "algorithm":"policy"})[["env", "policy", "seed", "step", "return"]]
    ppo_td3_df["seed"] = ppo_td3_df["seed"].fillna(0).astype(int)   
    ppo_td3_df = ppo_td3_df[(ppo_td3_df["env"].isin(['Hopper-v5', 'HalfCheetah-v5', 'Walker2d-v5', 'Humanoid-v5', 'Ant-v5']))&(ppo_td3_df["policy"].isin(["ppo","td3"]))&(ppo_td3_df["step"]<=1e6)]
    sac_df = pd.read_csv(baseline_logs_path+"sac_eval.csv").rename(columns={"eval_reward":"return","algo":"policy"})[["env", "policy", "seed", "step", "return"]]
    sac_df = sac_df[(sac_df["env"].isin(['Hopper-v5', 'HalfCheetah-v5', 'Walker2d-v5', 'Humanoid-v5', 'Ant-v5']))&(sac_df["step"]<=1e6)&(sac_df["seed"].isin([2, 12, 22, 32, 42]))]
    added_df = pd.read_csv(baseline_logs_path+"sb3_newExps.csv").rename(columns={"environment":"env", "eval_mean_reward":"return", "algorithm":"policy","epoch":"step"})[["env", "policy", "seed", "step", "return"]]
    added_df = added_df.groupby(["env", "policy", "seed"]).filter(lambda group: group["step"].max() == 1e6)
    added_df["env"] = added_df["env"].replace({"FrankaKitchen-N1-v1":"FrankaKitchen-v1-1", "FrankaKitchen-N2-v1":"FrankaKitchen-v1-2", "FrankaKitchen-N4-v1":"FrankaKitchen-v1-4"})
    tradition_df = pd.concat([ppo_td3_df, sac_df, added_df], ignore_index=True)
    tradition_df = tradition_df.groupby(["env", "policy", "seed"]).agg({"return": "max"})

    ########################################
    # Putting everything together and print the final table
    df_eval = pd.DataFrame(eval_results)
    df_eval = pd.concat([df_eval, tradition_df.reset_index()], ignore_index=True)
    df_eval.to_csv(f"{results_save_folder}evaluation.csv", index=False)

    ########################################
    # D4RL normalization:
    # - Adroit: overwrite return with normalized score
    # - Kitchen: keep raw return, and add normalized score in `success rate`
    adroit_mask = df_eval["env"].isin(adroit_envs_order)
    if adroit_mask.any():
        adroit_random = df_eval.loc[adroit_mask, "env"].map(lambda e: D4RL_SCORES[e]["random"])
        adroit_expert = df_eval.loc[adroit_mask, "env"].map(lambda e: D4RL_SCORES[e]["expert"])
        df_eval.loc[adroit_mask, "return"] = (df_eval.loc[adroit_mask, "return"] - adroit_random) / (adroit_expert - adroit_random) * 100.0

    df_eval["success_rate"] = np.nan
    kitchen_mask = df_eval["env"].isin(kitchen_envs_order)
    if kitchen_mask.any():
        kitchen_random = df_eval.loc[kitchen_mask, "env"].map(lambda e: D4RL_SCORES[e]["random"])
        kitchen_expert = df_eval.loc[kitchen_mask, "env"].map(lambda e: D4RL_SCORES[e]["expert"])
        df_eval.loc[kitchen_mask, "success_rate"] = (df_eval.loc[kitchen_mask, "return"] - kitchen_random) / (kitchen_expert - kitchen_random) * 100.0
    
    ########################################
    # Prepare for print
    mean_std = df_eval.groupby(["env", "policy"]).agg({"return": ["mean", "std"], "success_rate": ["mean", "std"]}).reset_index()
    mean_std.columns = ["env", "policy", "return_mean", "return_std", "success_rate_mean", "success_rate_std"]
    mean_std["return"] = mean_std.apply(lambda r: f"{r['return_mean']:.2f} ({r['return_std']:.2f})", axis=1)
    mean_std["success_rate"] = mean_std.apply(lambda r: f"{r['success_rate_mean']:.2f} ({r['success_rate_std']:.2f})", axis=1)
    mean_std = mean_std[["env", "policy", "return", "success_rate"]]

    def print_table_with_ampersand(table):
        print(table)
        for _, row in table.iterrows():
            print(" & ".join(row.fillna("-").astype(str).tolist()))

    mujoco_table = pd.DataFrame({"env": mujoco_envs_order}).merge(pd.DataFrame({"policy": policy_order}), how="cross")
    mujoco_table = mujoco_table.merge(mean_std, on=["env", "policy"], how="left")
    mujoco_table = mujoco_table.pivot(index="policy", columns="env", values="return").reindex(index=policy_order, columns=mujoco_envs_order).reset_index()
    print("**"*12+"\nMujoco Environments:")
    print_table_with_ampersand(mujoco_table)

    adroit_table = pd.DataFrame({"env": adroit_envs_order}).merge(pd.DataFrame({"policy": policy_order}), how="cross")
    adroit_table = adroit_table.merge(mean_std, on=["env", "policy"], how="left")
    adroit_table = adroit_table.pivot(index="policy", columns="env", values="return").reindex(index=policy_order, columns=adroit_envs_order).reset_index()
    print("**"*12+"\nAdroit Environments:")
    print_table_with_ampersand(adroit_table)

    kitchen_table = pd.DataFrame({"env": kitchen_envs_order}).merge(pd.DataFrame({"policy": policy_order}), how="cross")
    kitchen_table = kitchen_table.merge(mean_std, on=["env", "policy"], how="left")
    kitchen_table_return = kitchen_table.pivot(index="policy", columns="env", values="return").reindex(index=policy_order, columns=kitchen_envs_order).reset_index()
    print("**"*12+"\nKitchen Environments Rewards:")
    print_table_with_ampersand(kitchen_table_return)
    kitchen_table_acc = kitchen_table.pivot(index="policy", columns="env", values="success_rate").reindex(index=policy_order, columns=kitchen_envs_order).reset_index()
    print("**"*12+"\nKitchen Environments Success Rate:")
    print_table_with_ampersand(kitchen_table_acc)

def read_training_curve():
    api = wandb.Api()
    runs = api.runs("MyDiffusionRL/results")

    ########################################
    # Read results for diffusion-based baselines
    all_runs = []
    for run in runs:
        policy = run.name.split("_")[0]
        if policy not in policy_order:
            policy = "qvpo"
            env = run.name.split("_")[0]
            history = run.history(keys=["train/episode_reward", "_step"]).rename(columns={"train/episode_reward": "return", "_step": "step"})
            history["seed"] = run.name.split("seed=")[-1][0]
        else:
            env = run.name.split("_")[1]
            history = run.history(keys=["sample/episode_return", "_step"]).rename(columns={"sample/episode_return": "return", "_step": "step"})
            history["seed"] = run.name.split("_s")[-1]
        history["policy"] = policy
        history["env"] = env
        all_runs.append(history)

    train_df = pd.concat(all_runs, ignore_index=True)
    max_step = 1e6
    bin_width = max(1, int(max_step / 100))
    train_df['step_bin'] = (train_df['step'] // bin_width) * bin_width
    print(train_df)

    ########################################
    # Read results for FMER

    ########################################
    # Read results for traditional baselines
    tradition_df = pd.read_csv(baseline_logs_path+"sb3_newExps.csv").rename(columns={"environment":"env", "eval_mean_reward":"return", "algorithm":"policy","epoch":"step"})[["env", "policy", "seed", "step", "return"]]
    tradition_df = tradition_df.groupby(["env", "policy", "seed"]).filter(lambda group: group["step"].max() == 1e6)
    tradition_df["env"] = tradition_df["env"].replace({"FrankaKitchen-N1-v1":"FrankaKitchen-v1-1", "FrankaKitchen-N2-v1":"FrankaKitchen-v1-2", "FrankaKitchen-N4-v1":"FrankaKitchen-v1-4"})

if __name__=="__main__":
    D4RL_SCORES = {
            "AdroitHandHammer-v1": {"random": -274.856578, "expert": 12794.134825156867},
            "AdroitHandDoor-v1":   {"random": -56.512833,  "expert": 2880.5693087298737},
            "AdroitHandPen-v1":    {"random": 96.262799,    "expert": 3076.8331017826877},
            "AdroitHandRelocate-v1":{"random": -6.425911,  "expert": 4233.877797728884},
            "FrankaKitchen-v1":    {"random": 0.0,      "expert": 7.0},
            "FrankaKitchen-v1-1":  {"random": 0.0,      "expert": 1.0},
            "FrankaKitchen-v1-2":  {"random": 0.0,      "expert": 2.0},
            "FrankaKitchen-v1-4":  {"random": 0.0,      "expert": 4.0}}

    baseline_logs_path = "baseline_logs/"
    qvpo_logs_path = "/home/tinggao/qvpo/results_save/"
    FMER_logs_path = "exp_logs/FMER/"
    results_save_folder = "results/"

    mujoco_envs_order = ['Hopper-v5', 'HalfCheetah-v5', 'Walker2d-v5', 'Humanoid-v5', 'Ant-v5', 'InvertedPendulum-v5', 'Reacher-v5', 'Swimmer-v5', 'Pusher-v5', 'InvertedDoublePendulum-v5']
    kitchen_envs_order = ['FrankaKitchen-v1-1', 'FrankaKitchen-v1-2', 'FrankaKitchen-v1-4','FrankaKitchen-v1']
    adroit_envs_order = ['AdroitHandDoor-v1', 'AdroitHandHammer-v1','AdroitHandPen-v1', 'AdroitHandRelocate-v1']
    policy_order = ["ppo", "sac", "td3", "fpo", "dipo", "dpmd", "qvpo", "fmer"]

    read_evaluation()
    # read_training_curve()
    