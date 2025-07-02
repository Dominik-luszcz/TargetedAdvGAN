import numpy as np
import pandas as pd
import torch
from datetime import datetime
import pytorch_forecasting as pf
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import torch.nn as nn
from AdversarialAttackClasses import *

from torch.utils.data import DataLoader

import os

SAMPLE_LENGTH = 400

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


def plot(df: pd.DataFrame, title: str, to_plot: list, output_file: str = '.'):

    # first lets plot the adjprc and the attack adjprc
    plt.figure(figsize=(16, 6))
    colours = ["black", "blue", "green", "red", "brown", "orange", "purple", "cyan", "magenta"]
    i = 0
    labels = ["adjprc", "FGSM", "BIM", "MI-FGSM", "SIM","TIM (Up)","TAR (Down)","C&W"]
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
            tar_bim = TargetedIterativeMethod(model, iterations=10, direction=1, margin=100, epsilon=eps)
            tar_bim_normal, tar_U_bim_adjprc, tar_bim_attack = tar_bim.attack(df)
            tar_bim = TargetedIterativeMethod(model, iterations=10, direction=-1, margin=100, epsilon=eps)
            tar_bim_normal_down, tar_D_bim_adjprc, tar_bim_attack_down = tar_bim.attack(df)

            cw = CW_Attack(model, iterations=int(100 * 2.5), epsilon=2.5, c=1, direction=-1, size_penalty=0.1)
            cw_normal, cw_adjprc, cw_attack = cw.attack(df)

            attack_df = pd.DataFrame()
            pred_padding = np.zeros((300, 1))

            attack_df["adjprc"] = df["adjprc"]
            attack_df["normal_pred"] = np.vstack((pred_padding, fgsm_normal[1].unsqueeze(-1).detach().numpy()))

            attack_df["fgsm_adprc"] = fgsm_adjprc.detach().numpy()
            attack_df["fgsm_pred"] = np.vstack((pred_padding, fgsm_attack[1].unsqueeze(-1).detach().numpy()))

            attack_df["bim_adprc"] = bim_adjprc.detach().numpy()
            attack_df["bim_pred"] = np.vstack((pred_padding, bim_attack[1].unsqueeze(-1).detach().numpy()))

            attack_df["mi_fgsm_adprc"] = mi_fgsm_adjprc.detach().numpy()
            attack_df["mi_fgsm_pred"] = np.vstack((pred_padding, mi_fgsm_attack[1].unsqueeze(-1).detach().numpy()))

            attack_df["stealthy_adprc"] = s_bim_adjprc.detach().numpy()
            attack_df["stealthy_pred"] = np.vstack((pred_padding, s_bim_attack[1].unsqueeze(-1).detach().numpy()))

            attack_df["tar_U_bim_adprc"] = tar_U_bim_adjprc.detach().numpy()
            attack_df["tar_U_bim_pred"] = np.vstack((pred_padding, tar_bim_attack[1].unsqueeze(-1).detach().numpy()))

            attack_df["tar_D_bim_adprc"] = tar_D_bim_adjprc.detach().numpy()
            attack_df["tar_D_bim_pred"] = np.vstack((pred_padding, tar_bim_attack_down[1].unsqueeze(-1).detach().numpy()))

            attack_df["cw_adprc"] = cw_adjprc.detach().numpy()
            attack_df["cw_pred"] = np.vstack((pred_padding, cw_attack[1].unsqueeze(-1).detach().numpy()))

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

            i += 1

def plot_dataframes(data_path, output_dir):
    initialize_directory(f"{output_dir}/AttackAdjprc")
    initialize_directory(f"{output_dir}/AttackPredictions")

    for entry in Path(data_path).iterdir():
        if entry.suffix == ".csv":
            df = pd.read_csv(entry)
            ticker = entry.name.split("_attackdf")[0]
            plot(df, title=f"Adjprc for different attacks on {ticker}", to_plot=["adjprc", "fgsm_adprc", "bim_adprc", "mi_fgsm_adprc",
                              "stealthy_adprc","tar_U_bim_adprc","tar_D_bim_adprc","cw_adprc"], 
                              output_file=f"{output_dir}/AttackAdjprc/{ticker}.png")
            df = df[300:]
            plot(df, title=f"Predictions for different attacks on {ticker}", to_plot=["normal_pred","fgsm_pred","bim_pred","mi_fgsm_pred","stealthy_pred","tar_U_bim_pred",
                              "tar_D_bim_pred","cw_pred"], 
                              output_file=f"{output_dir}/AttackPredictions/{ticker}.png")



if __name__ == '__main__':
    t1 = datetime.now()
    print(f"Started job at {t1}")

    #perform_adversarial_attack("SP500_AttackData_Full", mode=0, output_path='Attack_Outputs/full_recording')
    perform_adversarial_attack("SP500_AttackData_Full", mode=1, output_path='Attack_Outputs/first400')
    #perform_adversarial_attack("SP500_AttackData_Full", mode=2, output_path='Attack_Outputs/final500')

    #plot_dataframes('Attack_Outputs/full_recording', 'Attack_Outputs/full_recording')
    #plot_dataframes('Attack_Outputs/first500', 'Attack_Outputs/first500')
    #plot_dataframes('Attack_Outputs/final500', 'Attack_Outputs/final500')

    t2 = datetime.now()
    print(f"Finished job at {t2} with job duration {t2 - t1}")