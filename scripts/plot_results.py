import argparse
from pathlib import Path
import pandas as pd
import wandb
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

D4RL_SCORES = {
    "AdroitHandHammer-v1": {"random": -274.856578, "expert": 12794.134825156867},
    "AdroitHandDoor-v1":   {"random": -56.512833,  "expert": 2880.5693087298737},
    "AdroitHandPen-v1":    {"random": 96.262799,    "expert": 3076.8331017826877},
    "AdroitHandRelocate-v1":{"random": -6.425911,  "expert": 4233.877797728884},
    "FrankaKitchen-v1":    {"random": 0.0,      "expert": 7.0},
}

def read_evaluation(env_name, policy_name):
    if policy_name.lower() in ["sac", "dpmd", "dipo"]:
        log_path = Path(f"baseline_logs/{env_name}/")
        eval_file_name = "evaluation.csv"
        folders = list(log_path.glob(f"{policy_name}*"))
        if len(folders)==0:
            folders = [p for p in log_path.glob(f"*/*/{policy_name}*") if p.is_dir()]
        col_name = "return"
    elif policy_name.lower() in ["qvpo"]:
        log_path = Path(f"/home/tinggao/qvpo/results_save/{env_name}")
        folders = list(log_path.glob(f"*"))
        eval_file_name = "eval_rewards.csv"
        col_name = "mean_reward"
    elif policy_name.lower() in ["gvpo"]:
        log_path = Path(f"exp_logs/0310-A/{env_name}/")
        folders = list(log_path.glob(f"*"))
        eval_file_name = "evaluation.csv"
        col_name = "return"
    else:
        log_path = Path(f"logs/{env_name}/")
    
    print(f"Found {len(folders)} runs for Policy: {policy_name}, Env: {env_name}")
    if len(folders)<5 or len(folders)>5:
        print(f"{policy_name} | {env_name}, Not enough runs found for Policy. Found {len(folders)} runs.")
        return "-", "-"

    eval_returns, first_300k_returns, best_of_last_returns = [], [], []
    for folder in folders:
        eval_file = f"{folder}/{eval_file_name}"
        df = pd.read_csv(eval_file)
        max_step = df["step"].max()
        if max_step<1e6:
            print(f"{policy_name}, {env_name},Unfinished run with only {df['step'].max()} steps.")
            return "-", "-"
        if env_name in D4RL_SCORES:
            random_score = D4RL_SCORES[env_name]["random"]
            expert_score = D4RL_SCORES[env_name]["expert"]
            df[col_name] = (df[col_name] - random_score) / (expert_score - random_score) * 100.0
        eval_returns.append(df[col_name].max())
        first_300k_returns.append(df[df["step"]<=200001][col_name].max())

        # 直接提取最后 10% 阶段内的评估最高分，不再做任何多余的平滑
        threshold_step = max_step * 0.90
        df_last_10_percent = df[df["step"] >= threshold_step]
        best_of_last = df_last_10_percent[col_name].max()
        best_of_last_returns.append(best_of_last)

    mean_return = sum(eval_returns)/len(eval_returns)
    std_return = pd.Series(eval_returns).std()

    # 计算 "Best of Last 10%" 的均值与标准差 (这才是你要填入论文的数字)
    mean_best_last = sum(best_of_last_returns)/len(best_of_last_returns)
    std_best_last = pd.Series(best_of_last_returns).std()
    # mean_300k_return = sum(first_300k_returns)/len(first_300k_returns)
    # std_300k_return = pd.Series(first_300k_returns).std()
    # print(f"{policy_name} | {env_name}, First 300k steps: &{mean_300k_return:.0f} ({std_300k_return:.0f})")
    if np.isnan(mean_return) or np.isnan(std_return) or np.isnan(mean_best_last) or np.isnan(std_best_last):
        return "-", "-"
    return f"{mean_return:.2f} ({std_return:.2f})", f"{mean_best_last:.2f} ({std_best_last:.2f})"

def read_training_curves(env_name_list, policy_name):
    api = wandb.Api()
    entity = "MyDiffusionRL"
    if policy_name.lower() == "dpmd":
        sweep_id = "baseline/tx7paeih"
    elif policy_name.lower() == "dipo":
        sweep_id = "baseline/bt68ivul"
    elif policy_name.lower() == "qvpo":
        sweep_id = "baseline/kemporfp"
    elif policy_name.lower() == "gvpo":
        sweep_id = "GVPO/pxyfsp7z"
        sweep_id_Hopper = "GVPO/9m9cygkf"

    # get all runs for the given env and policy
    sweep = api.sweep(f"{entity}/{sweep_id}")
    df = []
    for env_name in env_name_list:
        if policy_name.lower() == "gvpo" and env_name == "Hopper-v5":
            sweep = api.sweep(f"{entity}/{sweep_id_Hopper}")
            pattern = f"tau0.1_beta5.0"
        elif policy_name.lower() == "gvpo" and env_name != "Hopper-v5":
            sweep = api.sweep(f"{entity}/{sweep_id}")
            pattern = f"{policy_name}_{env_name}_"
        elif policy_name.lower() == "qvpo":
            pattern = f"{env_name}"
        else:
            pattern = f"{policy_name}_{env_name}_"
        matching_runs = [run for run in sweep.runs if pattern in run.name]
        assert len(matching_runs)==5, f"Expected 5 runs but found {len(matching_runs)} for Policy: {policy_name}, Env: {env_name}"    
        if len(matching_runs)<5:
            print(f"Not enough runs found for Policy: {policy_name}, Env: {env_name}, found {len(matching_runs)} runs.")
            continue

        # Access run data
        for run in matching_runs:
            # Get history with specific metric
            if policy_name.lower() == "qvpo":
                history = run.history(keys=["train/episode_reward", "_step"])
                history = history.rename(columns={"train/episode_reward": "sample/episode_return"})
                history["seed"] = run.name.split("seed=")[-1][0]
            else:
                history = run.history(keys=["sample/episode_return", "_step"])
                history["seed"] = run.name.split("_s")[-1]
            history["policy"] = policy_name
            history["env"] = env_name
            df.append(history)
    combined_df = pd.concat(df, ignore_index=True)
    return combined_df


def read_train_table(env_name_list, policy_name_list):
    all_df = []
    for policy_name in policy_name_list:
        if policy_name=="sac":
            file_path = "baseline_logs/sac_ppo_td3_train.csv"
            seeds = [2, 12, 22, 32, 42]
            df = pd.read_csv(file_path)
            df = df.rename(columns={"algo": "policy", "train_reward": "sample/episode_return", "step":"_step"})
            df = df[(df["env"].isin(env_name_list)) & (df["policy"]==policy_name) & (df["seed"].isin(seeds))]
        all_df.append(df)
    return pd.concat(all_df, ignore_index=True)

def read_eval_table(env_name_list, policy_name_list):
    all_df = []
    for policy_name in policy_name_list:
        if policy_name=="sac":
            file_path = "baseline_logs/sac_eval.csv"
            seeds = [2, 12, 22, 32, 42]
            df = pd.read_csv(file_path)
            df = df.rename(columns={"algo": "policy", "eval_reward": "evaluate/episode_reward", "step":"_step"})
            df = df[(df["env"].isin(env_name_list)) & (df["policy"]==policy_name) & (df["seed"].isin(seeds))]
        elif policy_name=="ppo" or policy_name=="td3":
            file_path = "baseline_logs/ppo_td3_eval.csv"
            df = pd.read_csv(file_path)
            df["seed"] = df["seed"].fillna(0).astype(int)   
            df['step'] = df['step'].astype(int)
            df = df.rename(columns={"algorithm": "policy", "eval_mean_reward": "evaluate/episode_reward", "environment": "env", "step":"_step"})
            df = df[(df["env"].isin(env_name_list)) & (df["policy"]==policy_name)]

        max_step = 1000000
        threshold_step = max_step * 0.90
        df = df[(df["_step"] >= threshold_step)&(df["_step"] <= max_step)]
        df = df[df["_step"] <= max_step]
        best_df = df.groupby(["env", "seed", "policy"])["evaluate/episode_reward"].max().reset_index()
        avg_performance = best_df.groupby(["env", "policy"])["evaluate/episode_reward"].mean().reset_index()
        std_performance = best_df.groupby(["env", "policy"])["evaluate/episode_reward"].std().reset_index()
        merged_df = pd.merge(avg_performance, std_performance, on=["env", "policy"], suffixes=("_mean", "_std"))

        merged_df["evaluate/episode_reward"] = merged_df.apply(lambda row: f"{row['evaluate/episode_reward_mean']:.2f} ({row['evaluate/episode_reward_std']:.2f})", axis=1)
        score_df = merged_df[["env", "policy", "evaluate/episode_reward"]].copy()
        score_df = score_df.merge(pd.DataFrame(env_name_list, columns=["env"]), on="env", how="right")
        score_df["evaluate/episode_reward"] = score_df["evaluate/episode_reward"].fillna("-")
        print(policy_name + "&" + "&".join(score_df["evaluate/episode_reward"].tolist()))

        all_df.append(df)
    return pd.concat(all_df, ignore_index=True)

def read_eval_curves(env_name_list, policy_name):
    api = wandb.Api()
    entity = "MyDiffusionRL"
    if policy_name.lower() == "dpmd":
        sweep_id = "baseline/tx7paeih"
    elif policy_name.lower() == "dipo":
        sweep_id = "baseline/bt68ivul"
    elif policy_name.lower() == "qvpo":
        sweep_id = "baseline/kemporfp"
    elif policy_name.lower() == "gvpo":
        sweep_id = "GVPO/pxyfsp7z"
        
    # get all runs for the given env and policy
    sweep = api.sweep(f"{entity}/{sweep_id}")
    df = []
    for env_name in env_name_list:
        if policy_name.lower() == "qvpo":
            pattern = f"{env_name}"
        else:
            pattern = f"{policy_name}_{env_name}_"
        
        matching_runs = [run for run in sweep.runs if run.name.startswith(pattern)]
        if len(matching_runs)<5:
            print(f"Not enough runs found for Policy: {policy_name}, Env: {env_name}, found {len(matching_runs)} runs.")
            continue

        # Access run data
        for run in matching_runs:
            # Get history with specific metric
            if policy_name.lower() == "qvpo":
                history = run.history(keys=["eval/reward", "_step"])
                history = history.rename(columns={"eval/reward": "evaluate/episode_reward"})
                history["seed"] = run.name.split("seed=")[-1][0]
            else:
                history = run.history(keys=["evaluate/episode_reward", "_step"])
                history["seed"] = run.name.split("_s")[-1]
            history["policy"] = policy_name
            history["env"] = env_name
            df.append(history)
    combined_df = pd.concat(df, ignore_index=True)
    return combined_df

def plot_curves(df, y_axis='sample/episode_return', suffix="train", envs_list=["HalfCheetah-v5","Hopper-v5",  "Walker2d-v5", "Humanoid-v5"]):
    max_step = df['_step'].max()
    bin_width = max(1, int(max_step / 100))
    df['step_bin'] = (df['_step'] // bin_width) * bin_width
    df['policy'] = df['policy'].map({'qvpo':'QVPO', 'gvpo':'FMER', 'dpmd':'DPMD', 'dipo':'DIPO', 'sac':'SAC'})
    
    # Set policy order for legend (FMER first)
    policy_order = ['SAC','DIPO','QVPO',  'FMER',  'DPMD',]
    df['policy'] = pd.Categorical(df['policy'], categories=policy_order, ordered=True)
    
    # Sort so FMER is drawn last (appears on top)
    df = df.sort_values('policy', ascending=False)
    
    # --- 2. 绘图 ---
    envs = envs_list
    n_envs = len(envs)
    
    # Create a single figure with subplots in a row
    fig, axes = plt.subplots(1, n_envs, figsize=(6 * n_envs, 5))
    
    # If only one environment, axes is not a list
    if n_envs == 1:
        axes = [axes]
    
    handles, labels = None, None
    
    for idx, env in enumerate(envs):
        env_df = df[df['env'] == env]
        
        sns.lineplot(
            data=env_df, 
            x='step_bin',               
            y=y_axis, 
            hue='policy',              
            errorbar='sd',               
            estimator='mean',            
            ax=axes[idx],
            legend=(idx == 0)            # Create legend only for first plot
        )
        
        
        # Also adjust the legend line width if it's the first plot
        if idx == 0:
            legend = axes[idx].get_legend()
        
        axes[idx].set_title(f'{env}', fontsize=16)
        axes[idx].set_xlabel("Steps", fontsize=14)
        axes[idx].set_ylabel("Episode Reward" if idx == 0 else "", fontsize=14)
        axes[idx].tick_params(axis='both', labelsize=12)
        axes[idx].grid(alpha=0.3)
        
        # Get handles and labels from the first plot, then remove its legend
        if idx == 0:
            handles, labels = axes[idx].get_legend_handles_labels()
            axes[idx].get_legend().remove()
    
    # Add a single shared legend below all subplots
    if handles and labels:
        fig.legend(handles, labels, loc='lower center', ncol=len(labels), 
                  bbox_to_anchor=(0.5, -0.05), frameon=True, fontsize=16)
    
    # Adjust layout to make room for the legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    Path("plots").mkdir(exist_ok=True)
    plt.savefig(f"plots/all_envs_{suffix}.png", dpi=400, bbox_inches='tight')
    plt.close()  

def read_ablation_results(sweep_id, patterns, run_ids=None, label="Default"):
    # default
    api = wandb.Api()
    if run_ids is None:
        sweep = api.sweep(f"MyDiffusionRL/GVPO/{sweep_id}")
        matching_runs = [run for run in sweep.runs if patterns in run.name]
    else:
        all_runs = api.runs("MyDiffusionRL/GVPO")
        run_ids_only = [rid.split("/")[-1] for rid in run_ids]
        matching_runs = [run for run in all_runs if run.id in run_ids_only]
    
    print(f"Loaded {len(matching_runs)} runs")
    df = []
    if len(matching_runs)<5:
        print(f"Not enough runs found for Default.")
    for run in matching_runs:
        history = run.history(keys=["sample/episode_return", "_step"])
        history["seed"] = run.name.split("_s")[-1]
        history["label"] = label
        history["tau"] = run.config.get("tau", None)
        df.append(history)
    return pd.concat(df, ignore_index=True)

def plot_bs_curves(df1, df2, y_axis='sample/episode_return', suffix="train"):
    df1[y_axis] = pd.to_numeric(df1[y_axis], errors='coerce')
    df1['_step'] = pd.to_numeric(df1['_step'], errors='coerce')
    df1 = df1.dropna(subset=[y_axis, '_step'])
    
    max_step = df1['_step'].max()
    bin_width = max(1, int(max_step / 100))
    df1['step_bin'] = (df1['_step'] // bin_width) * bin_width
    
    # Define consistent color palette
    color_palette = {
        'Default': '#1f77b4',  # blue
        'Unnormalized Weighting': '#ff7f0e',  # orange
        'Hard Top-1 Selection': '#2ca02c',  # green
        10000.0: '#ff7f0e',  # red
        100000.0: '#2ca02c',  # purple
        500000.0: '#8c564b',  # brown
    }
    
    # Create a single figure
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.lineplot(
        data=df1, 
        x='step_bin',               
        y=y_axis, 
        hue='label',
        palette=color_palette,
        errorbar=None,
        estimator='mean',            
        ax=ax[0],
        legend=True
    )
    
    ax[0].set_xlabel("Steps", fontsize=14)
    ax[0].set_ylabel("Episode Reward", fontsize=14)
    ax[0].tick_params(axis='both', labelsize=12)
    ax[0].grid(alpha=0.3)
    
    # Place legend on the plot
    ax[0].legend(title='', fontsize=14, loc='upper left')

    df2['step_bin'] = (df2['_step'] // bin_width) * bin_width
    sns.lineplot(
        data=df2, 
        x='step_bin',               
        y=y_axis, 
        hue='label',
        palette=color_palette,
        errorbar=None,
        estimator='mean',            
        ax=ax[1],
        legend=True
    )
    ax[1].set_xlabel("Steps", fontsize=14)
    ax[1].set_ylabel("Episode Reward", fontsize=14)
    ax[1].tick_params(axis='both', labelsize=12)
    ax[1].grid(alpha=0.3)
    
    # Add vertical lines to show when alpha has stopped
    ax[1].axvline(x=1e4, color='#ff7f0e', linestyle='--', linewidth=1.5, alpha=0.7)
    ax[1].axvline(x=1e5, color='#2ca02c', linestyle='--', linewidth=1.5, alpha=0.7)
    ax[1].axvline(x=5e5, color='#8c564b', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Update legend labels for ax[1]
    handles, labels = ax[1].get_legend_handles_labels()
    label_map = {
        'Default': 'Default',
        '10000.0': 'End after 1e4',
        '100000.0': 'End after 1e5',
        '500000.0': 'End after 5e5',
        '10000': 'End after 1e4',
        '100000': 'End after 1e5',
        '500000': 'End after 5e5'
    }
    new_labels = [label_map.get(label, label) for label in labels]
    ax[1].legend(handles, new_labels, title='', fontsize=14, loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    Path("plots").mkdir(exist_ok=True)
    plt.savefig(f"plots/ablation_{suffix}.png", dpi=400, bbox_inches='tight')
    plt.close()  

def plot_abl_curves(df, y_axis='sample/episode_return', suffix="train"):
    df[y_axis] = pd.to_numeric(df[y_axis], errors='coerce')
    df['_step'] = pd.to_numeric(df['_step'], errors='coerce')
    df = df.dropna(subset=[y_axis, '_step'])
    
    max_step = df['_step'].max()
    bin_width = max(1, int(max_step / 100))
    df['step_bin'] = (df['_step'] // bin_width) * bin_width
    
    # Define consistent color palette
    color_palette = {
        0.1: '#1f77b4',  # blue
        0.2: '#ff7f0e',  # orange
        0.4: '#2ca02c',  # green
    }
    
    # Create a single figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    
    sns.lineplot(
        data=df, 
        x='step_bin',               
        y=y_axis, 
        hue='tau',
        palette=color_palette,
        errorbar=None,
        estimator='mean',            
        ax=ax,
        legend=True
    )
    
    ax.set_xlabel("Steps", fontsize=14)
    ax.set_ylabel("Episode Reward", fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(alpha=0.3)
    
    # Place legend on the plot
    ax.legend(title='', fontsize=14, loc='upper left')
    
    # Update legend labels for ax   
    handles, labels = ax.get_legend_handles_labels()
    label_map = {
            "0.1": r'$\tau=0.1$',
            "0.2": r'$\tau=0.2$',
            "0.4": r'$\tau=0.4$',
    }
    new_labels = [label_map.get(label, label) for label in labels]
    ax.legend(handles, new_labels, title='', fontsize=14, loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    Path("plots").mkdir(exist_ok=True)
    plt.savefig(f"plots/ablation_{suffix}.png", dpi=400, bbox_inches='tight')
    plt.close()  

if __name__=="__main__":
    print("Reading evaluation results...")
    env_name_list = ["Hopper-v5", "HalfCheetah-v5", "Walker2d-v5","Humanoid-v5", "Ant-v5",  "InvertedPendulum-v5", "Reacher-v5", "Swimmer-v5", "Pusher-v5", "InvertedDoublePendulum-v5", "AdroitHandDoor-v1", "AdroitHandHammer-v1", "AdroitHandPen-v1", "AdroitHandRelocate-v1", "FrankaKitchen-v1"]
    # df = read_eval_table(env_name_list, policy_name_list=["ppo","sac","td3"])
    for policy_name in ["gvpo"]:
        mean, last = [], []
        for env_name in env_name_list:
            a, b = read_evaluation(env_name, policy_name)
            mean.append(a)
            last.append(b)
        print("**"*20)
        print(policy_name + " |  &" + " & ".join(mean[:10]))
        print(policy_name + " |  &" + " & ".join(last[:10]))
        print(policy_name + " |  &" + " & ".join(mean[10:]))
        print(policy_name + " |  &" + " & ".join(last[10:]))

    exit()
    
    # print("**"*20)
    # print("Reading training curves...")
    # all_curves_df = []
    # env_name_list = ["Hopper-v5", "HalfCheetah-v5", "Walker2d-v5", "Humanoid-v5"] #"Ant-v5", , "InvertedPendulum-v5"
    # for policy_name in ["qvpo","dpmd","dipo", "gvpo"]:
    #     all_curves_df.append(read_training_curves(env_name_list, policy_name))
    # all_curves_df.append(read_train_table(env_name_list, policy_name_list=["sac"]))
    # plot_curves(pd.concat(all_curves_df, ignore_index=True))
    
    # print("**"*20)
    # print("Reading evaluation curves...")
    # all_curves_df = []
    # env_name_list = ["Hopper-v5", "HalfCheetah-v5", "Walker2d-v5", "Humanoid-v5"] #"Ant-v5", , "InvertedPendulum-v5"
    # for policy_name in ["qvpo","dpmd","dipo", "gvpo"]:
    #     all_curves_df.append(read_eval_curves(env_name_list, policy_name))
    # all_curves_df.append(read_eval_table(env_name_list, policy_name_list=["sac"]))
    # plot_curves(pd.concat(all_curves_df, ignore_index=True), y_axis='evaluate/episode_reward', suffix="eval")

    # print("**"*20)
    # print("Reading ablation studies...")
    # # ablation study 1
    # df0 = read_ablation_results(sweep_id="pxyfsp7z", patterns="gvpo_Walker2d-v5", label="Default")
    # df1 = read_ablation_results(sweep_id="sbvmxliw", patterns="gvpo_Walker2d-v5", label="Unnormalized Weighting")
    # df2 = read_ablation_results(sweep_id="6z3urp6q", patterns="gvpo_Walker2d-v5", label="Hard Top-1 Selection")
    # bs1_df = pd.concat([df0, df1, df2], ignore_index=True)

    # #  ablation study 2
    # run_ids1=["MyDiffusionRL/GVPO/rn4qjsxn", 
    #         "MyDiffusionRL/GVPO/dzawsmr5",
    #         "MyDiffusionRL/GVPO/qghdyjea",
    #         "MyDiffusionRL/GVPO/ubf0o2u2",
    #         "MyDiffusionRL/GVPO/yklovvjb"]
    # df1 = read_ablation_results(sweep_id=None, patterns=None, run_ids=run_ids1, label=1e4)

    # run_ids2=["MyDiffusionRL/GVPO/0f7jcyt1",
    #         "MyDiffusionRL/GVPO/3cnfy73l",
    #         "MyDiffusionRL/GVPO/6mzz6ytp",
    #         "MyDiffusionRL/GVPO/w8m4gewh",
    #         "MyDiffusionRL/GVPO/zsi988ln"]
    # df2 = read_ablation_results(sweep_id=None, patterns=None, run_ids=run_ids2, label=1e5)


    # run_ids3=["MyDiffusionRL/GVPO/k5wybgoe",
    #         "MyDiffusionRL/GVPO/ld83gzjv",
    #         "MyDiffusionRL/GVPO/q8ldka09",
    #         "MyDiffusionRL/GVPO/rie8yvmj",
    #         "MyDiffusionRL/GVPO/ut46jn3q"]
    # df3 = read_ablation_results(sweep_id=None, patterns=None, run_ids=run_ids3, label=5e5)  
    # bs2_df = pd.concat([df0, df1, df2, df3], ignore_index=True)

    # plot_bs_curves(bs1_df, bs2_df, y_axis='sample/episode_return', suffix="")


    # # print("**"*20)
    # print("Reading ablation studies on tau...")
    # df = read_ablation_results(sweep_id="v4478bdg", patterns="beta4.0", label="Tau Ablation")
    # plot_abl_curves(df, y_axis='sample/episode_return', suffix="tau_ablation")
    # print(df)