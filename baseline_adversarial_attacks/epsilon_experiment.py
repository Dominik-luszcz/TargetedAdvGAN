import numpy as np
import pandas as pd
import torch
from datetime import datetime
import pytorch_forecasting as pf
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
)
import matplotlib.pyplot as plt
import torch.nn as nn
from AdversarialAttackClasses import *
from collections import defaultdict

from torch.utils.data import DataLoader

import os

SAMPLE_LENGTH = 400

ATTACK_MODES = {
    "full_recording": 0,
    "random_sample": 1,
    "recent_history": 2,
}


def initialize_directory(path: str) -> None:
    """Create the output folder at path if it does not exist, or empty it if it exists."""
    if os.path.exists(path):
        for filename in os.listdir(path):
            os.remove(os.path.join(path, filename))
    else:
        os.mkdir(path)


def plot_figure(normal, attack, ticker, eps, method="FGSM", output_path="."):
    plt.figure(figsize=(14, 6))
    plt.plot(normal[1].detach().numpy(), label="Normal Predictions", color="blue")
    plt.plot(attack[1].detach().numpy(), label="Attack Predictions", color="orange")
    plt.title(
        f"{method} for {ticker}, Normal MAE = {normal[0]}, Attack MAE = {attack[0]} (eps = {eps})"
    )
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("adjprc")
    plt.savefig(f"{output_path}/{ticker}_{method}.png")
    plt.close()


def plot(df: pd.DataFrame, title: str, to_plot: list, output_file: str = "."):

    # first lets plot the adjprc and the attack adjprc
    plt.figure(figsize=(16, 6))
    colours = [
        "black",
        "blue",
        "green",
        "red",
        "brown",
        "orange",
        "purple",
        "cyan",
        "magenta",
    ]
    i = 0
    for prc in to_plot:
        if i == 0:
            plt.plot(df[prc], label=prc, color=colours[i], linewidth=3, zorder=15)
        else:
            plt.plot(df[prc], label=prc, color=colours[i])
        i += 1
    plt.title(title)
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("adjprc")
    plt.savefig(output_file)
    plt.close()


def get_epsilon(df: pd.DataFrame):
    adjprc = torch.from_numpy(df["adjprc"].to_numpy())

    # calculate mean returns for our epsilon
    ret = torch.abs(adjprc[1:] / adjprc[:-1])
    return torch.mean(ret)


def run_baseline_attacks(data_path, mode=0, output_path=".", eps=0):
    model_state_dict = torch.load("NHITS_forecasting_model.pt")
    params = torch.load("./NHITS_params.pt", weights_only=False)
    params["loss"] = pf.QuantileLoss(
        quantiles=[0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]
    )
    model = pf.NHiTS(**params)
    model.load_state_dict(model_state_dict)
    model.eval()
    total_normal_error = 0
    total_attack_error = 0
    k = 0

    mae_experiment = []

    i = 0

    # For each recording we perform the adversarial attack
    for entry in Path(data_path).iterdir():
        if i > 5:
            break
        if entry.suffix == ".csv":
            if mode == 0:
                df = pd.read_csv(entry)
            if mode == 1:
                df = pd.read_csv(entry).head(SAMPLE_LENGTH)
            if mode == 2:
                df = pd.read_csv(entry).tail(SAMPLE_LENGTH)
                df = df.reset_index(drop=True)

            # eps = get_epsilon(df)

            # 1. FGSM
            fgsm = FGSM(model, epsilon=eps)
            fgsm_normal, fgsm_adjprc, fgsm_attack = fgsm.attack(df)

            # 2. BIM
            bim = BasicIterativeMethod(model, iterations=10, epsilon=eps)
            bim_normal, bim_adjprc, bim_attack = bim.attack(df)

            # 3. MI-FGSM
            mi_fgsm = MI_FGSM(model, iterations=10, decay=0.35, epsilon=eps)
            mi_fgsm_normal, mi_fgsm_adjprc, mi_fgsm_attack = mi_fgsm.attack(df)

            # 4. Stealthy BIM
            s_bim = StealthyIterativeMethod(model, iterations=10, epsilon=eps)
            s_bim_normal, s_bim_adjprc, s_bim_attack = s_bim.attack(df)

            # 5. Targeted BIM
            tar_bim = TargetedIterativeMethod(
                model, iterations=10, direction=1, margin=100, epsilon=eps
            )
            tar_bim_normal, tar_U_bim_adjprc, tar_bim_attack = tar_bim.attack(df)
            tar_bim = TargetedIterativeMethod(
                model, iterations=10, direction=-1, margin=100, epsilon=eps
            )
            tar_bim_normal_down, tar_D_bim_adjprc, tar_bim_attack_down = tar_bim.attack(
                df
            )

            cw = CW_Attack(
                model,
                iterations=int(100 * eps),
                epsilon=eps,
                c=1,
                direction=-1,
                size_penalty=0.1,
            )
            cw_normal, cw_adjprc, cw_attack = cw.attack(df)

            attack_df = pd.DataFrame()
            pred_padding = np.zeros((300, 1))

            attack_df["adjprc"] = df["adjprc"]
            attack_df["normal_pred"] = np.vstack(
                (pred_padding, fgsm_normal[1].unsqueeze(-1).detach().numpy())
            )

            attack_df["fgsm_adprc"] = fgsm_adjprc.detach().numpy()
            attack_df["fgsm_pred"] = np.vstack(
                (pred_padding, fgsm_attack[1].unsqueeze(-1).detach().numpy())
            )

            attack_df["bim_adprc"] = bim_adjprc.detach().numpy()
            attack_df["bim_pred"] = np.vstack(
                (pred_padding, bim_attack[1].unsqueeze(-1).detach().numpy())
            )

            attack_df["mi_fgsm_adprc"] = mi_fgsm_adjprc.detach().numpy()
            attack_df["mi_fgsm_pred"] = np.vstack(
                (pred_padding, mi_fgsm_attack[1].unsqueeze(-1).detach().numpy())
            )

            attack_df["stealthy_adprc"] = s_bim_adjprc.detach().numpy()
            attack_df["stealthy_pred"] = np.vstack(
                (pred_padding, s_bim_attack[1].unsqueeze(-1).detach().numpy())
            )

            attack_df["tar_U_bim_adprc"] = tar_U_bim_adjprc.detach().numpy()
            attack_df["tar_U_bim_pred"] = np.vstack(
                (pred_padding, tar_bim_attack[1].unsqueeze(-1).detach().numpy())
            )

            attack_df["tar_D_bim_adprc"] = tar_D_bim_adjprc.detach().numpy()
            attack_df["tar_D_bim_pred"] = np.vstack(
                (pred_padding, tar_bim_attack_down[1].unsqueeze(-1).detach().numpy())
            )

            attack_df["cw_adprc"] = cw_adjprc.detach().numpy()
            attack_df["cw_pred"] = np.vstack(
                (pred_padding, cw_attack[1].unsqueeze(-1).detach().numpy())
            )

            attack_df.to_csv(
                f"{output_path}/{entry.name.split(".csv")[0]}_attackdf.csv", index=False
            )

            print(f"Completed: {entry}")

            # i += 1


def plot_dataframes(data_path, output_dir):
    initialize_directory(f"{output_dir}/AttackAdjprc")
    initialize_directory(f"{output_dir}/AttackPredictions")

    for entry in Path(data_path).iterdir():
        if entry.suffix == ".csv":
            df = pd.read_csv(entry)
            ticker = entry.name.split("attackdf")[0]
            plot(
                df,
                title=f"Adjprc for different attacks on {ticker}",
                to_plot=[
                    "adjprc",
                    "fgsm_adprc",
                    "bim_adprc",
                    "mi_fgsm_adprc",
                    "stealthy_adprc",
                    "tar_U_bim_adprc",
                    "tar_D_bim_adprc",
                    "cw_adprc",
                ],
                output_file=f"{output_dir}/AttackAdjprc/{ticker}.png",
            )
            df = df[300:]
            plot(
                df,
                title=f"Predictions for different attacks on {ticker}",
                to_plot=[
                    "normal_pred",
                    "fgsm_pred",
                    "bim_pred",
                    "mi_fgsm_pred",
                    "stealthy_pred",
                    "tar_U_bim_pred",
                    "tar_D_bim_pred",
                    "cw_pred",
                ],
                output_file=f"{output_dir}/AttackPredictions/{ticker}.png",
            )


def epsilon_experiment(
    data_path,
    output_dir,
    mode,
    epsilons=[0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5],
):

    for eps in epsilons:
        output_path = f"{output_dir}/Epsilon_{eps}"
        initialize_directory(output_path)
        run_baseline_attacks(
            data_path=data_path, mode=mode, output_path=output_path, eps=eps
        )


def bar_plot(data_path, epsilon, output_dir="."):

    maes = {
        "Normal": 0,
        "FGSM": 0,
        "BIM": 0,
        "MI-FGSM": 0,
        "Stealthy": 0,
        "TIM-Up": 0,
        "TIM-Down": 0,
        "C&W": 0,
    }

    rsmes = {
        "Normal": 0,
        "FGSM": 0,
        "BIM": 0,
        "MI-FGSM": 0,
        "Stealthy": 0,
        "TIM-Up": 0,
        "TIM-Down": 0,
        "C&W": 0,
    }

    mapes = {
        "Normal": 0,
        "FGSM": 0,
        "BIM": 0,
        "MI-FGSM": 0,
        "Stealthy": 0,
        "TIM-Up": 0,
        "TIM-Down": 0,
        "C&W": 0,
    }

    i = 0
    for entry in Path(data_path).iterdir():
        if entry.suffix != ".csv" or "attackdf" not in entry.name:
            continue

        df = pd.read_csv(entry)
        df = df.tail(100)
        maes["Normal"] += mean_absolute_error(df["adjprc"], df["normal_pred"])
        maes["FGSM"] += mean_absolute_error(df["adjprc"], df["fgsm_pred"])
        maes["BIM"] += mean_absolute_error(df["adjprc"], df["bim_pred"])
        maes["MI-FGSM"] += mean_absolute_error(df["adjprc"], df["mi_fgsm_pred"])
        maes["Stealthy"] += mean_absolute_error(df["adjprc"], df["stealthy_pred"])
        maes["TIM-Up"] += mean_absolute_error(df["adjprc"], df["tar_U_bim_pred"])
        maes["TIM-Down"] += mean_absolute_error(df["adjprc"], df["tar_D_bim_pred"])
        maes["C&W"] += mean_absolute_error(df["adjprc"], df["cw_pred"])

        rsmes["Normal"] += root_mean_squared_error(df["adjprc"], df["normal_pred"])
        rsmes["FGSM"] += root_mean_squared_error(df["adjprc"], df["fgsm_pred"])
        rsmes["BIM"] += root_mean_squared_error(df["adjprc"], df["bim_pred"])
        rsmes["MI-FGSM"] += root_mean_squared_error(df["adjprc"], df["mi_fgsm_pred"])
        rsmes["Stealthy"] += root_mean_squared_error(df["adjprc"], df["stealthy_pred"])
        rsmes["TIM-Up"] += root_mean_squared_error(df["adjprc"], df["tar_U_bim_pred"])
        rsmes["TIM-Down"] += root_mean_squared_error(df["adjprc"], df["tar_D_bim_pred"])
        rsmes["C&W"] += root_mean_squared_error(df["adjprc"], df["cw_pred"])

        mapes["Normal"] += mean_absolute_percentage_error(
            df["adjprc"], df["normal_pred"]
        )
        mapes["FGSM"] += mean_absolute_percentage_error(df["adjprc"], df["fgsm_pred"])
        mapes["BIM"] += mean_absolute_percentage_error(df["adjprc"], df["bim_pred"])
        mapes["MI-FGSM"] += mean_absolute_percentage_error(
            df["adjprc"], df["mi_fgsm_pred"]
        )
        mapes["Stealthy"] += mean_absolute_percentage_error(
            df["adjprc"], df["stealthy_pred"]
        )
        mapes["TIM-Up"] += mean_absolute_percentage_error(
            df["adjprc"], df["tar_U_bim_pred"]
        )
        mapes["TIM-Down"] += mean_absolute_percentage_error(
            df["adjprc"], df["tar_D_bim_pred"]
        )
        mapes["C&W"] += mean_absolute_percentage_error(df["adjprc"], df["cw_pred"])

        i += 1

    values = np.array(list(maes.values())) / i

    labels = maes.keys()
    colors = ["black", "red", "blue", "green", "purple", "magenta", "cyan", "brown"]
    plt.bar(labels, values, color=colors)
    plt.xlabel("Prediction")
    plt.ylabel("Mean Absolute Error")
    plt.title("Avg MAE over Different Attack Methods")
    plt.savefig(f"{output_dir}/avg_attack_mae_{epsilon}.png")
    plt.close()

    return maes, rsmes, mapes, i


def plot_epsilon_experiment_bar_graph(
    experiment_maes, output_dir=".", num_recordings=1
):

    eps = list(experiment_maes.keys())
    labels = list(list(experiment_maes.values())[0].keys())
    num_labels = len(labels)

    bar_width = 0.25
    spacing = 1.5

    group_label_positions = []
    current_position = 0

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["black", "red", "blue", "green", "purple", "grey", "orange", "brown"]

    for i, e in enumerate(eps):
        values = np.array(list(experiment_maes[e].values())) / num_recordings

        position = current_position + np.arange(num_labels) * bar_width
        ax.bar(position, values, bar_width, label=labels, color=colors)

        group_center = current_position + ((bar_width * num_labels) / 2 - bar_width / 2)
        group_label_positions.append(group_center)
        current_position += spacing + num_labels * bar_width

    ax.set_xticks(group_label_positions)
    ax.set_xticklabels(eps)
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title("Avg MAE Over Different Epsilon Values")
    ax.legend(labels=labels)
    plt.savefig(f"{output_dir}/epsilon_experiment.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    t1 = datetime.now()
    print(f"Started job at {t1}")

    eps = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5]
    experiment_maes = {}
    for e in eps:
        output_dir = (
            f"Attack_Outputs/EpsilonExperiments2/first{SAMPLE_LENGTH}/Epsilon_{e}"
        )
        maes, rsmes, mapes, num_recordings = bar_plot(
            output_dir, e, output_dir=output_dir
        )
        experiment_maes[f"{e}"] = maes

        temp = maes
        values = np.array(list(maes.values())) / num_recordings
        keys = list(maes.keys())
        for i in range(len(keys)):
            temp[keys[i]] = values[i]

        df = pd.DataFrame(temp, index=[0])
        df.to_csv(f"{output_dir}/{e}_eps_maes.csv")

        temp = rsmes
        values = np.array(list(rsmes.values())) / num_recordings
        keys = list(rsmes.keys())
        for i in range(len(keys)):
            temp[keys[i]] = values[i]

        df = pd.DataFrame(temp, index=[0])
        df.to_csv(f"{output_dir}/{e}_eps_rsmes.csv")

        temp = mapes
        values = np.array(list(mapes.values())) / num_recordings
        keys = list(mapes.keys())
        for i in range(len(keys)):
            temp[keys[i]] = values[i]

        df = pd.DataFrame(temp, index=[0])
        df.to_csv(f"{output_dir}/{e}_eps_mapes.csv")

    plot_epsilon_experiment_bar_graph(
        experiment_maes=experiment_maes,
        output_dir=f"Attack_Outputs/EpsilonExperiments2/first{SAMPLE_LENGTH}",
        num_recordings=1,
    )

    # bar_plot(f'Attack_Outputs/EpsilonExperiment/final{SAMPLE_LENGTH}/Epsilon_{e}', e, output_dir=f'Attack_Outputs/EpsilonExperiment/final{SAMPLE_LENGTH}/Epsilon_{e}')

    # output_path = f'Attack_Outputs/EpsilonExperiments2/first{SAMPLE_LENGTH}'
    # initialize_directory(output_path)
    # epsilon_experiment("SP500_AttackData_Full", mode=1, output_dir=output_path, epsilons=eps)

    # output_path = f'Attack_Outputs/EpsilonExperiment/final{SAMPLE_LENGTH}'
    # initialize_directory(output_path)
    # epsilon_experiment("SP500_AttackData_Full", mode=2, output_dir=output_path, epsilons=eps)

    # perform_adversarial_attack("SP500_AttackData_Full", mode=0, output_path='Attack_Outputs/full_recording')
    # perform_adversarial_attack("SP500_AttackData_Full", mode=1, output_path='Attack_Outputs/first500')
    # perform_adversarial_attack("SP500_AttackData_Full", mode=2, output_path='Attack_Outputs/final500')

    # plot_dataframes('Attack_Outputs/full_recording', 'Attack_Outputs/full_recording')
    # plot_dataframes('Attack_Outputs/first500', 'Attack_Outputs/first500')
    # plot_dataframes('Attack_Outputs/final500', 'Attack_Outputs/final500')

    t2 = datetime.now()
    print(f"Finished job at {t2} with job duration {t2 - t1}")
