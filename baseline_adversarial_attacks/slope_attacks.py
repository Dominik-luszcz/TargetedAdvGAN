import numpy as np
import pandas as pd
import torch
from datetime import datetime
import pytorch_forecasting as pf
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix
import matplotlib.pyplot as plt
import torch.nn as nn
from AdversarialAttackClasses import *

from torch.utils.data import DataLoader

import os

SAMPLE_LENGTH = 120

ATTACK_MODES = {
    'full_recording': 0,
    'random_sample': 1,
    'recent_history': 2,
}

def initialize_directory(path: str) -> None:
    """Create the output folder at path if it does not exist, or empty it if it exists."""
    if os.path.exists(path):
        for filename in os.listdir(path):
            os.remove(os.path.join(path, filename))
    else:
        os.mkdir(path)

def plot_figure(normal, attack, ticker, eps, method = 'FGSM', output_path = '.'):
    plt.figure(figsize=(14, 6))
    plt.plot(normal[1].detach().numpy(), label="Normal Predictions", color="blue")
    plt.plot(attack[1].detach().numpy(), label="Attack Predictions", color="orange")
    plt.title(f"{method} for {ticker}, Normal MAE = {normal[0]}, Attack MAE = {attack[0]} (eps = {eps})")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("adjprc")
    plt.savefig(f'{output_path}/{ticker}_{method}.png')
    plt.close()


def plot(df: pd.DataFrame, title: str, to_plot: list, output_file: str = '.',
         labels = ["adjprc", "FGSM", "BIM", "MI-FGSM", "SIM","TIM (Up)","TAR (Down)","C&W", "Slope (Up)", "Slope (Down)", "Slope (Zero)"]):

    # first lets plot the adjprc and the attack adjprc
    plt.figure(figsize=(16, 6))
    colours = ["black", "blue", "green", "red", "cyan", "magenta", "pink", "grey", "brown", "orange", "purple",]
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


def get_epsilon(df: pd.DataFrame):
    adjprc = torch.from_numpy(df["adjprc"].to_numpy())

    # calculate mean returns for our epsilon
    ret = torch.abs(adjprc[1:] / adjprc[:-1])
    return torch.mean(ret)

def perform_adversarial_attack(data_path, mode=0, output_path = '.'):
    model_state_dict = torch.load("NHITS_forecasting_model.pt")
    params = torch.load("./NHITS_params.pt", weights_only=False)
    params["loss"] = pf.QuantileLoss(quantiles=[0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999])
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

            eps = get_epsilon(df)
            eps = 5

            try:
                slope = Slope_Attack(model, iterations=10, target_direction=1, c = 5, d = 2, epsilon=eps)
                slope_normal_up, slope_up_adjprc, slope_attack_up = slope.attack(df)

                slope = Slope_Attack(model, iterations=10, target_direction=-1, c = 5, d = 2, epsilon=eps)
                slope_normal_down, slope_down_adjprc, slope_attack_down = slope.attack(df)

                slope = Slope_Attack(model, iterations=10, target_direction=0, c = 5, d = 2, epsilon=eps)
                slope_normal_0, slope_0_adjprc, slope_attack_0 = slope.attack(df)

                ls_slope = LS_Slope_Attack(model, iterations=10, target_direction=1, c = 5, d = 2, epsilon=eps)
                ls_slope_normal_up, ls_slope_up_adjprc, ls_slope_attack_up = ls_slope.attack(df)

                ls_slope = LS_Slope_Attack(model, iterations=10, target_direction=-1, c = 5, d = 2, epsilon=eps)
                ls_slope_normal_down, ls_slope_down_adjprc, ls_slope_attack_down = ls_slope.attack(df)

                ls_slope = LS_Slope_Attack(model, iterations=10, target_direction=0, c = 5, d = 2, epsilon=eps)
                ls_slope_normal_0, ls_slope_0_adjprc, ls_slope_attack_0 = ls_slope.attack(df)
            except:
                continue


            # cw_slope = CW_BasicSlope_Attack(model, iterations=int(100 * 2.5), target_direction=1, c = 500,d = 2, epsilon=eps)
            # cw_slope_normal_up, cw_slope_up_adjprc, cw_slope_attack_up = cw_slope.attack(df)

            # cw_slope = CW_BasicSlope_Attack(model, iterations=int(100 * 2.5), target_direction=-1, c = 500,d = 2, epsilon=eps)
            # cw_slope_normal_down, cw_slope_down_adjprc, cw_slope_attack_down = cw_slope.attack(df)

            # cw_slope = CW_BasicSlope_Attack(model, iterations=int(100 * 2.5), target_direction=0, c = 2000,d = 2, epsilon=eps)
            # cw_slope_normal_0, cw_slope_0_adjprc, cw_slope_attack_0 = cw_slope.attack(df)

            # cw_ls_slope = CW_LS_Attack(model, iterations=int(100 * 2.5), target_direction=1, c = 700,d = 2, epsilon=eps)
            # cw_ls_slope_normal_up, cw_ls_slope_up_adjprc, cw_ls_slope_attack_up = cw_ls_slope.attack(df)

            # cw_ls_slope = CW_LS_Attack(model, iterations=int(100 * 2.5), target_direction=-1, c = 500,d = 2, epsilon=eps)
            # cw_ls_slope_normal_down, cw_ls_slope_down_adjprc, cw_ls_slope_attack_down = cw_ls_slope.attack(df)

            # cw_ls_slope = CW_LS_Attack(model, iterations=int(100 * 2.5), target_direction=0, c = 2000,d = 2, epsilon=eps)
            # cw_ls_slope_normal_0, cw_ls_slope_0_adjprc, cw_ls_slope_attack_0 = cw_ls_slope.attack(df)
            

            attack_df = pd.DataFrame()
            pred_padding = np.zeros((100, 1))

            attack_df["adjprc"] = df["adjprc"]
            attack_df["normal_pred"] = np.vstack((pred_padding, ls_slope_normal_0[1].unsqueeze(-1).detach().numpy()))

            attack_df["slope_up_adjprc"] = slope_up_adjprc.detach().numpy()
            attack_df["slope_up_pred"] = np.vstack((pred_padding, slope_attack_up[1].unsqueeze(-1).detach().numpy()))

            attack_df["slope_down_adjprc"] = slope_down_adjprc.detach().numpy()
            attack_df["slope_down_pred"] = np.vstack((pred_padding, slope_attack_down[1].unsqueeze(-1).detach().numpy()))

            attack_df["slope_0_adjprc"] = slope_0_adjprc.detach().numpy()
            attack_df["slope_0_pred"] = np.vstack((pred_padding, slope_attack_0[1].unsqueeze(-1).detach().numpy()))

            attack_df["ls_slope_up_adjprc"] = ls_slope_up_adjprc.detach().numpy()
            attack_df["ls_slope_up_pred"] = np.vstack((pred_padding, ls_slope_attack_up[1].unsqueeze(-1).detach().numpy()))

            attack_df["ls_slope_down_adjprc"] = ls_slope_down_adjprc.detach().numpy()
            attack_df["ls_slope_down_pred"] = np.vstack((pred_padding, ls_slope_attack_down[1].unsqueeze(-1).detach().numpy()))

            attack_df["ls_slope_0_adjprc"] = ls_slope_0_adjprc.detach().numpy()
            attack_df["ls_slope_0_pred"] = np.vstack((pred_padding, ls_slope_attack_0[1].unsqueeze(-1).detach().numpy()))

            # attack_df["cw_slope_up_adjprc"] = cw_slope_up_adjprc.detach().numpy()
            # attack_df["cw_slope_up_pred"] = np.vstack((pred_padding, cw_slope_attack_up[1].unsqueeze(-1).detach().numpy()))

            # attack_df["cw_slope_down_adjprc"] = cw_slope_down_adjprc.detach().numpy()
            # attack_df["cw_slope_down_pred"] = np.vstack((pred_padding, cw_slope_attack_down[1].unsqueeze(-1).detach().numpy()))

            # attack_df["cw_slope_0_adjprc"] = cw_slope_0_adjprc.detach().numpy()
            # attack_df["cw_slope_0_pred"] = np.vstack((pred_padding, cw_slope_attack_0[1].unsqueeze(-1).detach().numpy()))

            # attack_df["cw_ls_slope_up_adjprc"] = cw_ls_slope_up_adjprc.detach().numpy()
            # attack_df["cw_ls_slope_up_pred"] = np.vstack((pred_padding, cw_ls_slope_attack_up[1].unsqueeze(-1).detach().numpy()))

            # attack_df["cw_ls_slope_down_adjprc"] = cw_ls_slope_down_adjprc.detach().numpy()
            # attack_df["cw_ls_slope_down_pred"] = np.vstack((pred_padding, cw_ls_slope_attack_down[1].unsqueeze(-1).detach().numpy()))

            # attack_df["cw_ls_slope_0_adjprc"] = cw_ls_slope_0_adjprc.detach().numpy()
            # attack_df["cw_ls_slope_0_pred"] = np.vstack((pred_padding, cw_ls_slope_attack_0[1].unsqueeze(-1).detach().numpy()))
            


            attack_df.to_csv(f"{output_path}/{entry.name.split(".csv")[0]}_attackdf.csv", index=False)


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

            #i += 1

def plot_dataframes(data_path, output_dir):
    # initialize_directory(f"{output_dir}/AttackAdjprc")
    # initialize_directory(f"{output_dir}/AttackPredictions")
    initialize_directory(f"{output_dir}/Slope_Attacks")
    initialize_directory(f"{output_dir}/Slope_Adjprc")
    # initialize_directory(f"{output_dir}/CW_Slope_Attacks")
    # initialize_directory(f"{output_dir}/CW_Slope_Adjprc")

    for entry in Path(data_path).iterdir():
        if entry.suffix == ".csv":
            df = pd.read_csv(entry)
            ticker = entry.name.split("_attackdf")[0]
            # plot(df, title=f"Adjprc for different attacks on {ticker}", to_plot=["adjprc", "fgsm_adprc", "bim_adprc", "mi_fgsm_adprc",
            #                   "stealthy_adprc","tar_U_bim_adprc","tar_D_bim_adprc", "cw_adprc", "slope_up_adjprc", "slope_down_adjprc", "slope_0_adjprc"], 
            #                   output_file=f"{output_dir}/AttackAdjprc/{ticker}.png")
            
            plot(df, title=f"Adjprc for different attacks on {ticker}", to_plot=["adjprc", "slope_up_adjprc", "slope_down_adjprc", "slope_0_adjprc", "ls_slope_up_adjprc", "ls_slope_down_adjprc", "ls_slope_0_adjprc"], 
                              output_file=f"{output_dir}/Slope_Adjprc/{ticker}.png", labels=['Adjprc', 'Slope (Up)', 'Slope (Down)', 'Slope (0)', 'LS Slope (Up)', 'LS Slope (Down)', 'LS Slope (0)'])
            
            # plot(df, title=f"Adjprc for different attacks on {ticker}", to_plot=["adjprc", "cw_slope_up_adjprc", "cw_slope_down_adjprc", "cw_slope_0_adjprc", "cw_ls_slope_up_adjprc", "cw_ls_slope_down_adjprc", "cw_ls_slope_0_adjprc"], 
            #                   output_file=f"{output_dir}/CW_Slope_Adjprc/{ticker}.png", labels=['Adjprc', 'CW Slope (Up)', 'CW Slope (Down)', 'CW Slope (0)', 'CW LS Slope (Up)', 'CW LS Slope (Down)', 'CW LS Slope (0)'])
            
            df = df[100:]
            # plot(df, title=f"Predictions for different attacks on {ticker}", to_plot=["normal_pred","fgsm_pred","bim_pred","mi_fgsm_pred","stealthy_pred","tar_U_bim_pred",
            #                   "tar_D_bim_pred","cw_pred", "slope_up_pred", "slope_down_pred", "slope_0_pred"], 
            #                   output_file=f"{output_dir}/AttackPredictions/{ticker}.png")
            
            plot(df, title=f"Predictions for slope attacks on {ticker}", to_plot=["normal_pred", "slope_up_pred", "slope_down_pred", "slope_0_pred", "ls_slope_up_pred", "ls_slope_down_pred", "ls_slope_0_pred"], 
                              output_file=f"{output_dir}/Slope_Attacks/{ticker}.png", labels=['Normal Pred', 'Slope (Up)', 'Slope (Down)', 'Slope (0)', 'LS Slope (Up)', 'LS Slope (Down)', 'LS Slope (0)'])
            
            # plot(df, title=f"Predictions for slope attacks on {ticker}", to_plot=["normal_pred", "cw_slope_up_pred", "cw_slope_down_pred", "cw_slope_0_pred", "cw_ls_slope_up_pred", "cw_ls_slope_down_pred", "cw_ls_slope_0_pred"], 
            #         output_file=f"{output_dir}/CW_Slope_Attacks/{ticker}.png", labels=['Normal Pred', 'CW Slope (Up)', 'CW Slope (Down)', 'CW Slope (0)', 'CW LS Slope (Up)', 'CW LS Slope (Down)', 'CW LS Slope (0)'])




if __name__ == '__main__':
    t1 = datetime.now()
    print(f"Started job at {t1}")

    #perform_adversarial_attack("SP500_AttackData_Full", mode=0, output_path='Attack_Outputs/full_recording')
    perform_adversarial_attack("SP500_Filtered", mode=1, output_path='Attack_Outputs/first120_slope_bim_eps5')
    #perform_adversarial_attack("SP500_AttackData_Full", mode=2, output_path='Attack_Outputs/final500')

    #plot_dataframes('Attack_Outputs/full_recording', 'Attack_Outputs/full_recording')
    #plot_dataframes('Attack_Outputs/first500', 'Attack_Outputs/first500')
    #plot_dataframes('Attack_Outputs/final500', 'Attack_Outputs/final500')


    plot_dataframes('Attack_Outputs/first120_slope_bim_eps5', 'Attack_Outputs/first120_slope_bim_eps5')

    t2 = datetime.now()
    print(f"Finished job at {t2} with job duration {t2 - t1}")