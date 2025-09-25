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
import json

from torch.utils.data import DataLoader

import os

SAMPLE_LENGTH = 300

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


def plot(
    df: pd.DataFrame,
    title: str,
    to_plot: list,
    output_file: str = ".",
    labels=[
        "adjprc",
        "FGSM",
        "BIM",
        "MI-FGSM",
        "SIM",
        "TIM (Up)",
        "TAR (Down)",
        "C&W",
        "Slope (Up)",
        "Slope (Down)",
        "Slope (Zero)",
    ],
):

    # first lets plot the adjprc and the attack adjprc
    plt.figure(figsize=(16, 6))
    colours = [
        "black",
        "blue",
        "green",
        "red",
        "cyan",
        "magenta",
        "pink",
        "grey",
        "brown",
        "orange",
        "purple",
    ]
    i = 0
    for prc in to_plot:
        if i == 0:
            plt.plot(df[prc], label=labels[i], color=colours[i], linewidth=3, zorder=15)
        else:
            plt.plot(df[prc], label=labels[i], color=colours[i])
        i += 1
    plt.title(title)
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("adjprc")
    plt.savefig(output_file)
    plt.close()


def get_epsilon(df: pd.DataFrame, percentage):
    adjprc = torch.from_numpy(df["adjprc"].to_numpy())

    return percentage * torch.median(adjprc)

    # # calculate mean returns for our epsilon
    # ret = torch.abs(adjprc[1:] - adjprc[:-1])
    # return torch.mean(ret)


def perform_adversarial_attack(data_path, percentage, mode=0, output_path="."):
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

    # output_path = "Attack_Outputs"
    initialize_directory(output_path)

    mae_experiment = []
    # initialize_directory(f"{output_path}/StealthyBIM")
    # initialize_directory(f"{output_path}/MIFGSM")
    # initialize_directory(f"{output_path}/BIM")
    # initialize_directory(f"{output_path}/FGSM")
    # initialize_directory(f"{output_path}/TarBIM_up")
    # initialize_directory(f"{output_path}/TarBIM_down")
    # initialize_directory(f"{output_path}/CW")

    i = 0
    output_dir_items = Path(output_path).iterdir()
    output_dir_items = [entry.name for entry in output_dir_items]

    # For each recording we perform the adversarial attack
    for entry in Path(data_path).iterdir():
        if f"{entry.name.split(".")[0]}_attackdf.csv" in output_dir_items:
            continue
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

            eps = get_epsilon(df, percentage)

            try:

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
                tar_bim_normal_down, tar_D_bim_adjprc, tar_bim_attack_down = (
                    tar_bim.attack(df)
                )

                cw = CW_Attack(
                    model,
                    iterations=int(100 * 2.5),
                    epsilon=2.5,
                    c=1,
                    direction=-1,
                    size_penalty=0.1,
                    clamp=True,
                )
                cw_normal, cw_adjprc, cw_attack = cw.attack(df)

                slope = Slope_Attack(
                    model, iterations=10, target_direction=1, c=5, d=2, epsilon=eps
                )
                slope_normal_up, slope_up_adjprc, slope_attack_up = slope.attack(df)

                slope = Slope_Attack(
                    model, iterations=10, target_direction=-1, c=5, d=2, epsilon=eps
                )
                slope_normal_down, slope_down_adjprc, slope_attack_down = slope.attack(
                    df
                )

                slope = Slope_Attack(
                    model, iterations=10, target_direction=0, c=5, d=2, epsilon=eps
                )
                slope_normal_0, slope_0_adjprc, slope_attack_0 = slope.attack(df)

                ls_slope = LS_Slope_Attack(
                    model, iterations=10, target_direction=1, c=5, d=2, epsilon=eps
                )
                ls_slope_normal_up, ls_slope_up_adjprc, ls_slope_attack_up = (
                    ls_slope.attack(df)
                )

                ls_slope = LS_Slope_Attack(
                    model, iterations=10, target_direction=-1, c=5, d=2, epsilon=eps
                )
                ls_slope_normal_down, ls_slope_down_adjprc, ls_slope_attack_down = (
                    ls_slope.attack(df)
                )

                ls_slope = LS_Slope_Attack(
                    model, iterations=10, target_direction=0, c=5, d=2, epsilon=eps
                )
                ls_slope_normal_0, ls_slope_0_adjprc, ls_slope_attack_0 = (
                    ls_slope.attack(df)
                )

                cw_slope = CW_BasicSlope_Attack(
                    model,
                    iterations=int(100 * 2.5),
                    target_direction=1,
                    c=500,
                    d=2,
                    epsilon=eps,
                )
                cw_slope_normal_up, cw_slope_up_adjprc, cw_slope_attack_up = (
                    cw_slope.attack(df)
                )

                cw_slope = CW_BasicSlope_Attack(
                    model,
                    iterations=int(100 * 2.5),
                    target_direction=-1,
                    c=500,
                    d=2,
                    epsilon=eps,
                )
                cw_slope_normal_down, cw_slope_down_adjprc, cw_slope_attack_down = (
                    cw_slope.attack(df)
                )

                cw_slope = CW_BasicSlope_Attack(
                    model,
                    iterations=int(100 * 2.5),
                    target_direction=0,
                    c=2000,
                    d=2,
                    epsilon=eps,
                )
                cw_slope_normal_0, cw_slope_0_adjprc, cw_slope_attack_0 = (
                    cw_slope.attack(df)
                )

                cw_ls_slope = CW_LS_Attack(
                    model,
                    iterations=int(100 * 2.5),
                    target_direction=1,
                    c=700,
                    d=2,
                    epsilon=eps,
                )
                cw_ls_slope_normal_up, cw_ls_slope_up_adjprc, cw_ls_slope_attack_up = (
                    cw_ls_slope.attack(df)
                )

                cw_ls_slope = CW_LS_Attack(
                    model,
                    iterations=int(100 * 2.5),
                    target_direction=-1,
                    c=500,
                    d=2,
                    epsilon=eps,
                )
                (
                    cw_ls_slope_normal_down,
                    cw_ls_slope_down_adjprc,
                    cw_ls_slope_attack_down,
                ) = cw_ls_slope.attack(df)

                cw_ls_slope = CW_LS_Attack(
                    model,
                    iterations=int(100 * 2.5),
                    target_direction=0,
                    c=2000,
                    d=2,
                    epsilon=eps,
                )
                cw_ls_slope_normal_0, cw_ls_slope_0_adjprc, cw_ls_slope_attack_0 = (
                    cw_ls_slope.attack(df)
                )

                attack_df = pd.DataFrame()
                pred_padding = np.zeros((100, 1))

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
                    (
                        pred_padding,
                        tar_bim_attack_down[1].unsqueeze(-1).detach().numpy(),
                    )
                )

                attack_df["cw_adprc"] = cw_adjprc.detach().numpy()
                attack_df["cw_pred"] = np.vstack(
                    (pred_padding, cw_attack[1].unsqueeze(-1).detach().numpy())
                )

                attack_df["slope_up_adjprc"] = slope_up_adjprc.detach().numpy()
                attack_df["slope_up_pred"] = np.vstack(
                    (pred_padding, slope_attack_up[1].unsqueeze(-1).detach().numpy())
                )

                attack_df["slope_down_adjprc"] = slope_down_adjprc.detach().numpy()
                attack_df["slope_down_pred"] = np.vstack(
                    (pred_padding, slope_attack_down[1].unsqueeze(-1).detach().numpy())
                )

                attack_df["slope_0_adjprc"] = slope_0_adjprc.detach().numpy()
                attack_df["slope_0_pred"] = np.vstack(
                    (pred_padding, slope_attack_0[1].unsqueeze(-1).detach().numpy())
                )

                attack_df["ls_slope_up_adjprc"] = ls_slope_up_adjprc.detach().numpy()
                attack_df["ls_slope_up_pred"] = np.vstack(
                    (pred_padding, ls_slope_attack_up[1].unsqueeze(-1).detach().numpy())
                )

                attack_df["ls_slope_down_adjprc"] = (
                    ls_slope_down_adjprc.detach().numpy()
                )
                attack_df["ls_slope_down_pred"] = np.vstack(
                    (
                        pred_padding,
                        ls_slope_attack_down[1].unsqueeze(-1).detach().numpy(),
                    )
                )

                attack_df["ls_slope_0_adjprc"] = ls_slope_0_adjprc.detach().numpy()
                attack_df["ls_slope_0_pred"] = np.vstack(
                    (pred_padding, ls_slope_attack_0[1].unsqueeze(-1).detach().numpy())
                )

                attack_df["cw_slope_up_adjprc"] = cw_slope_up_adjprc.detach().numpy()
                attack_df["cw_slope_up_pred"] = np.vstack(
                    (pred_padding, cw_slope_attack_up[1].unsqueeze(-1).detach().numpy())
                )

                attack_df["cw_slope_down_adjprc"] = (
                    cw_slope_down_adjprc.detach().numpy()
                )
                attack_df["cw_slope_down_pred"] = np.vstack(
                    (
                        pred_padding,
                        cw_slope_attack_down[1].unsqueeze(-1).detach().numpy(),
                    )
                )

                attack_df["cw_slope_0_adjprc"] = cw_slope_0_adjprc.detach().numpy()
                attack_df["cw_slope_0_pred"] = np.vstack(
                    (pred_padding, cw_slope_attack_0[1].unsqueeze(-1).detach().numpy())
                )

                attack_df["cw_ls_slope_up_adjprc"] = (
                    cw_ls_slope_up_adjprc.detach().numpy()
                )
                attack_df["cw_ls_slope_up_pred"] = np.vstack(
                    (
                        pred_padding,
                        cw_ls_slope_attack_up[1].unsqueeze(-1).detach().numpy(),
                    )
                )

                attack_df["cw_ls_slope_down_adjprc"] = (
                    cw_ls_slope_down_adjprc.detach().numpy()
                )
                attack_df["cw_ls_slope_down_pred"] = np.vstack(
                    (
                        pred_padding,
                        cw_ls_slope_attack_down[1].unsqueeze(-1).detach().numpy(),
                    )
                )

                attack_df["cw_ls_slope_0_adjprc"] = (
                    cw_ls_slope_0_adjprc.detach().numpy()
                )
                attack_df["cw_ls_slope_0_pred"] = np.vstack(
                    (
                        pred_padding,
                        cw_ls_slope_attack_0[1].unsqueeze(-1).detach().numpy(),
                    )
                )

                attack_df.to_csv(
                    f"{output_path}/{entry.name.split(".csv")[0]}_attackdf.csv",
                    index=False,
                )

                # plot_figure(normal = s_bim_normal, attack = s_bim_attack, ticker = entry.name.split(".csv")[0], eps = eps,
                #             method='StealthyBIM', output_path=f"{output_path}/StealthyBIM")

                # plot_figure(normal = mi_fgsm_normal, attack = mi_fgsm_attack, ticker = entry.name.split(".csv")[0], eps = eps,
                #             method='MIFGSM', output_path=f"{output_path}/MIFGSM")

                # plot_figure(normal = bim_normal, attack = bim_attack, ticker = entry.name.split(".csv")[0], eps = eps,
                #             method='BIM', output_path=f"{output_path}/BIM")

                # plot_figure(normal = fgsm_normal, attack = fgsm_attack, ticker = entry.name.split(".csv")[0], eps = eps,
                #             method='FGSM', output_path=f"{output_path}/FGSM")

                # plot_figure(normal = tar_bim_normal, attack = tar_bim_attack, ticker = entry.name.split(".csv")[0], eps = eps,
                #             method='TarBIM', output_path=f"{output_path}/TarBIM_up")
                # plot_figure(normal = tar_bim_normal_down, attack = tar_bim_attack_down, ticker = entry.name.split(".csv")[0], eps = eps,
                #             method='TarBIM', output_path=f"{output_path}/TarBIM_down")

                # plot_figure(normal = cw_normal, attack = cw_attack, ticker = entry.name.split(".csv")[0], eps = eps,
                #             method='CW', output_path=f"{output_path}/CW")

                print(f"Completed: {entry}")
            except:
                print(f"Problem with {entry}. Going to skip")

            i += 1


def plot_dataframes(data_path, output_dir):
    initialize_directory(f"{output_dir}/AttackAdjprc")
    initialize_directory(f"{output_dir}/AttackPredictions")
    initialize_directory(f"{output_dir}/Slope_Attacks")
    initialize_directory(f"{output_dir}/Slope_Adjprc")
    initialize_directory(f"{output_dir}/CW_Slope_Attacks")
    initialize_directory(f"{output_dir}/CW_Slope_Adjprc")

    for entry in Path(data_path).iterdir():
        if entry.suffix == ".csv":
            df = pd.read_csv(entry)
            ticker = entry.name.split("_attackdf")[0]
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

            plot(
                df,
                title=f"Adjprc for different attacks on {ticker}",
                to_plot=[
                    "adjprc",
                    "slope_up_adjprc",
                    "slope_down_adjprc",
                    "slope_0_adjprc",
                    "ls_slope_up_adjprc",
                    "ls_slope_down_adjprc",
                    "ls_slope_0_adjprc",
                ],
                output_file=f"{output_dir}/Slope_Adjprc/{ticker}.png",
                labels=[
                    "Adjprc",
                    "GSA (Up)",
                    "GSA (Down)",
                    "GSA (0)",
                    "LSSA (Up)",
                    "LSSA (Down)",
                    "LSSA (0)",
                ],
            )

            # plot(df, title=f"Adjprc for different attacks on {ticker}", to_plot=["adjprc", "cw_slope_up_adjprc", "cw_slope_down_adjprc", "cw_slope_0_adjprc", "cw_ls_slope_up_adjprc", "cw_ls_slope_down_adjprc", "cw_ls_slope_0_adjprc"],
            #                   output_file=f"{output_dir}/CW_Slope_Adjprc/{ticker}.png", labels=['Adjprc', 'CW GSA (Up)', 'CW GSA (Down)', 'CW GSA (0)', 'CW LSSA (Up)', 'CW LSSA (Down)', 'CW LSSA (0)'])

            df = df[100:]
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

            plot(
                df,
                title=f"Predictions for slope attacks on {ticker}",
                to_plot=[
                    "normal_pred",
                    "slope_up_pred",
                    "slope_down_pred",
                    "slope_0_pred",
                    "ls_slope_up_pred",
                    "ls_slope_down_pred",
                    "ls_slope_0_pred",
                ],
                output_file=f"{output_dir}/Slope_Attacks/{ticker}.png",
                labels=[
                    "Normal Pred",
                    "GSA (Up)",
                    "GSA (Down)",
                    "GSA (0)",
                    "LSSA (Up)",
                    "LSSA (Down)",
                    "LSSA (0)",
                ],
            )

            # plot(df, title=f"Predictions for slope attacks on {ticker}", to_plot=["normal_pred", "cw_slope_up_pred", "cw_slope_down_pred", "cw_slope_0_pred", "cw_ls_slope_up_pred", "cw_ls_slope_down_pred", "cw_ls_slope_0_pred"],
            #         output_file=f"{output_dir}/CW_Slope_Attacks/{ticker}.png", labels=['Normal Pred', 'CW GSA (Up)', 'CW GSA (Down)', 'CW GSA (0)', 'CW LSSA (Up)', 'CW LSSA (Down)', 'CW LSSA (0)'])


def get_attack_metrics(folder, columns, ground_truth, index_start, output_dir):

    mae_dict = dict.fromkeys(columns, 0)
    rmse_dict = dict.fromkeys(columns, 0)
    mape_dict = dict.fromkeys(columns, 0)
    gs_dict = dict.fromkeys(columns, 0)
    lss_dict = dict.fromkeys(columns, 0)

    i = 0
    for entry in Path(folder).iterdir():
        if entry.suffix == ".csv":
            df = pd.read_csv(entry)
            df = df.tail(len(df) - index_start)
            i += 1
            for c in columns:
                mae_dict[c] += mean_absolute_error(df[ground_truth], df[c])
                rmse_dict[c] += root_mean_squared_error(df[ground_truth], df[c])
                mape_dict[c] += mean_absolute_percentage_error(df[ground_truth], df[c])

                npy_c = df[c].to_numpy()
                gs = (npy_c[-1] - npy_c[0]) / len(npy_c)
                gs_dict[c] += gs

                x = np.arange(len(npy_c))
                x_mean = np.mean(x)
                y_mean = np.mean(npy_c)

                numerator = ((x - x_mean) * (npy_c - y_mean)).sum()
                denom = ((x - x_mean) ** 2).sum()

                lss = numerator / denom
                lss_dict[c] += lss

    mae_dict = {k: v / i for k, v in mae_dict.items()}
    rmse_dict = {k: v / i for k, v in rmse_dict.items()}
    mape_dict = {k: v / i for k, v in mape_dict.items()}
    gs_dict = {k: v / i for k, v in gs_dict.items()}
    lss_dict = {k: v / i for k, v in lss_dict.items()}

    with open(f"{output_dir}/mae_dict.json", "w") as f:
        json.dump(mae_dict, f, indent=2)

    with open(f"{output_dir}/rmse_dict.json", "w") as f:
        json.dump(rmse_dict, f, indent=2)

    with open(f"{output_dir}/mape_dict.json", "w") as f:
        json.dump(mape_dict, f, indent=2)

    with open(f"{output_dir}/gs_dict.json", "w") as f:
        json.dump(gs_dict, f, indent=2)

    with open(f"{output_dir}/lss_dict.json", "w") as f:
        json.dump(lss_dict, f, indent=2)


if __name__ == "__main__":
    t1 = datetime.now()
    print(f"Started job at {t1}")

    # percentages = [0.005]
    # for p in percentages:

    #     get_attack_metrics(f"Attack_Outputs/first400_relative_eps_{p}",
    #                     columns=["normal_pred", "fgsm_pred", "bim_pred",
    #                                 "mi_fgsm_pred", "stealthy_pred", "tar_U_bim_pred",
    #                                 "tar_D_bim_pred", "cw_pred", "slope_up_pred",
    #                                 "slope_down_pred", "slope_0_pred", "ls_slope_up_pred",
    #                                     "ls_slope_down_pred", "ls_slope_0_pred"],
    #                         ground_truth="adjprc", index_start=100,
    #                         output_dir=f"Attack_Outputs/first400_relative_eps_{p}")

    max_eps = 0
    min_eps = 1000
    for entry in Path("SP500_AttackData_full").iterdir():
        if entry.suffix == ".csv":
            df = pd.read_csv(entry)
            df = df.head(120)
            print(df["adjprc"].median())

            eps = get_epsilon(df, 0.02)
            if eps > max_eps:
                max_eps = eps
            if eps < min_eps:
                min_eps = eps

    print(f"Max Eps: {max_eps}")
    print(f"Min Eps: {min_eps}")

    # percentages = [0.02]
    # for p in percentages:

    # #perform_adversarial_attack("SP500_AttackData_Full", mode=0, output_path='Attack_Outputs/full_recording')
    #     perform_adversarial_attack("SP500_AttackData_Full", p, mode=1, output_path=f'Attack_Outputs/first300_relative_eps_{p}_all_cw')
    # perform_adversarial_attack("SP500_AttackData_Full", mode=2, output_path='Attack_Outputs/final500')

    # plot_dataframes('Attack_Outputs/full_recording', 'Attack_Outputs/full_recording')
    # plot_dataframes('Attack_Outputs/first500', 'Attack_Outputs/first500')
    # plot_dataframes('Attack_Outputs/final500', 'Attack_Outputs/final500')

    plot_dataframes(
        "Attack_Outputs/first300_relative_0.02_cw",
        "Attack_Outputs/first300_relative_0.02_cw",
    )

    t2 = datetime.now()
    print(f"Finished job at {t2} with job duration {t2 - t1}")
