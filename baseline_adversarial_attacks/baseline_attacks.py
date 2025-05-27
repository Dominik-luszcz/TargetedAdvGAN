import numpy as np
import pandas as pd
import torch
from datetime import datetime
import pytorch_forecasting as pf
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import torch.nn as nn

from torch.utils.data import DataLoader

import os

SAMPLE_LENGTH = 500

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

        
'''
Compute the ema based on the pandas formula in their documentation (when adjust=False)
'''
def exponential_moving_average(adjprc: torch.Tensor, span: int):

    alpha = 2.0 / (span + 1.0)

    ema = torch.zeros_like(adjprc) 
    ema[0] = adjprc[0]

    # formula taken from the pandas definition
    for i in range(1, len(adjprc)):
        ema[i] = (1 - alpha) * adjprc[i-1] + alpha * adjprc[i]

    return ema

def feature_generation(adjprc: torch.Tensor):

    weight5 = torch.ones((1,1,5))/5
    weight10 = torch.ones((1,1,10))/10
    weight20 = torch.ones((1,1,20))/20
    rolling_mean5 = torch.nn.functional.conv1d(adjprc.unsqueeze(0).unsqueeze(0), weight5, stride=1)

    rolling_mean10 = torch.nn.functional.conv1d(adjprc.unsqueeze(0).unsqueeze(0), weight10, stride=1)

    rolling_mean20 = torch.nn.functional.conv1d(adjprc.unsqueeze(0).unsqueeze(0), weight20, stride=1)

    # stdevs: Variance = E[x^2] - E[x]^2
    # so E[x] = rolling_meanX => E[x]^2 = rolling_meanX ^ 2
    rollingExp_mean5 = torch.nn.functional.conv1d(adjprc.unsqueeze(0).unsqueeze(0) ** 2, weight5, stride=1)
    rollingExp_mean10 = torch.nn.functional.conv1d(adjprc.unsqueeze(0).unsqueeze(0) ** 2, weight10, stride=1)
    rollingExp_mean20 = torch.nn.functional.conv1d(adjprc.unsqueeze(0).unsqueeze(0) ** 2, weight20, stride=1)

    rolling_stdev5 = torch.sqrt(rollingExp_mean5 - rolling_mean5**2 + 1e-06)
    rolling_stdev10 = torch.sqrt(rollingExp_mean10 - rolling_mean10**2 + 1e-06)
    rolling_stdev20 = torch.sqrt(rollingExp_mean20 - rolling_mean20**2 + 1e-06)
    # but we have padding and the model fills the first x values with something (either adjprc or mean rolling_stdev) so we have to fix that
    rolling_mean5 = rolling_mean5.squeeze(0).squeeze(0)
    rolling_mean5 = torch.cat([adjprc[:4], rolling_mean5])
    rolling_stdev5 = rolling_stdev5.squeeze(0).squeeze(0)
    rolling_stdev5 = torch.cat([torch.full(size=[4], fill_value=torch.mean(rolling_stdev5.squeeze(0).squeeze(0)).item()), rolling_stdev5])

    rolling_mean10 = rolling_mean10.squeeze(0).squeeze(0)
    rolling_mean10 = torch.cat([adjprc[:9], rolling_mean10])
    rolling_stdev10 = rolling_stdev10.squeeze(0).squeeze(0)
    rolling_stdev10 = torch.cat([torch.full(size=[9], fill_value=torch.mean(rolling_stdev10.squeeze(0).squeeze(0)).item()), rolling_stdev10])

    rolling_mean20 = rolling_mean20.squeeze(0).squeeze(0)
    rolling_mean20 = torch.cat([adjprc[:19], rolling_mean20])
    rolling_stdev20 = rolling_stdev20.squeeze(0).squeeze(0)
    rolling_stdev20 = torch.cat([torch.full(size=[19], fill_value=torch.mean(rolling_stdev20.squeeze(0).squeeze(0)).item()), rolling_stdev20])

    # log returns 
    log_returns = torch.log(adjprc[1:]/adjprc[:-1]) 
    log_returns = torch.cat([torch.tensor([0.0]), log_returns]) 

    # ROC (percetn change of 5)
    roc5 = (adjprc[5:] - adjprc[:-5]) / adjprc[:-5]
    roc5 = torch.cat([torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]), roc5])

    # exponential moving averages
    ema_5 = exponential_moving_average(adjprc=adjprc, span=5)
    ema_10 = exponential_moving_average(adjprc=adjprc, span=10)
    ema_20 = exponential_moving_average(adjprc=adjprc, span=20)

    features = torch.stack([rolling_mean5, rolling_mean10, rolling_mean20, 
                             rolling_stdev5, rolling_stdev10, rolling_stdev20, 
                             log_returns, roc5, ema_5, ema_10, ema_20], dim=1)


    # Because we are doing 1 recording at a time, the standard scalar for teh features (not target) are just the mean and stdev of the recording
    # Similarly, the quantiles are the same for the target for the robust scalar for adjprc

    avg = torch.mean(features, dim=0)
    stdev = torch.std(features, dim=0)

    features = (features - avg) / stdev

    # median = torch.median(adjprc, dim=0).values
    # q75 = torch.quantile(adjprc, q=0.75, dim=0)
    # q25 = torch.quantile(adjprc, q=0.25, dim=0)
    # #adjprc = (adjprc - median) / (q75 - q25)

    # scale_ = q75 - q25
    # center_ = median

    center_ = np.median(adjprc.detach().numpy())
    scale_ = (np.quantile(adjprc.detach().numpy(), q=0.75) - np.quantile(adjprc.detach().numpy(), q=0.25)) / 2

    features = torch.hstack([adjprc.unsqueeze(-1), features])

    return features, scale_, center_
        

def call_model(projection_model: pf.NHiTS, adjprc: torch.Tensor, days):
    x_prime = adjprc
    
    features, scale_, center_ = feature_generation(x_prime)
    scale_ = torch.tensor(scale_)
    center_ = torch.tensor(center_)

    features[:, 0] = (features[:, 0] - center_) / scale_

    features = features.type(torch.float32)


    # craft payload
    # payload: (dictionary)
    # 1. encoder_cat -> long (batch_size x n_encoder_time_steps x n_features) -> long tensor of encoded categoricals for encoder
    # 2. encoder_cont -> float tensor of scaled continuous variables for encoder
    # 3. encoder_lengths -> long tensor with lengths of the encoder time series. No entry will be greater than n_encoder_time_steps
    # 4. encoder_target -> if list, each entry for a different target. float tensor with unscaled continous target or encoded categorical target, list of tensors for multiple targets
    # 5. decoder_cat -> long tensor of encoded categoricals for decoder
    # 6. decoder_cont -> float tensor of scaled continuous variables for decoder
    # 7. decoder_lengths -> long tensor with lengths of the decoder time series. No entry will be greater than n_decoder_time_steps
    # 8. decoder_target -> if list, with each entry for a different target. float tensor with unscaled continous target or encoded categorical target for decoder - this corresponds to first entry of y, list of tensors for multiple targets
    # 9. target_scale -> if list, with each entry for a different target. parameters used to normalize the target. Typically these are mean and standard deviation. Is list of tensors for multiple targets.

    encoder_batches = []
    encoder_categorical_batches = []
    decoder_batches = []
    decoder_categorical_batches = []
    time_idx = []
    encoder_targets = []
    decoder_targets = []

    for i in range(0, len(features) - 300 - 49): # need to make room for prediction values
        encoder_batches.append(features[i:i+300, :])
        encoder_categorical_batches.append(days[i:i+300])
        encoder_targets.append(x_prime[i:i+300])

        decoder_batches.append(features[i+300:i+300+50, :])
        decoder_categorical_batches.append(days[i+300 : i+300+50])
        decoder_targets.append(x_prime[i+300:i+350])
        time_idx.append(torch.arange(i+300, i+350))


    encoder_batches = torch.stack(encoder_batches)
    encoder_batches = encoder_batches.requires_grad_()
    encoder_categorical_batches = torch.stack(encoder_categorical_batches).unsqueeze(-1)

    encoder_targets = torch.stack(encoder_targets)


    decoder_batches = torch.stack(decoder_batches)
    decoder_categorical_batches = torch.stack(decoder_categorical_batches).unsqueeze(-1)

    decoder_targets = torch.stack(decoder_targets)
    
    # will need to batch this further
    payload = {
        'encoder_cat': encoder_categorical_batches, # this should be the categorical features from the lookback period
        'encoder_cont': encoder_batches, # this should be the continous features from the lookback period
        'encoder_lengths': torch.tensor([300]), # should be length of lookback period
        'encoder_target': encoder_targets, # should be the target variable of teh lookback period (adjprc)
        'decoder_cat': decoder_categorical_batches, # should be the categorical features of the next 50 features
        'decoder_cont': decoder_batches, # should be the continous features of the next 50 features
        'decoder_lengths': torch.tensor([50]), # should be the length of the prediction
        'decoder_target': decoder_targets, # should be the ground truths for the next 50 adjprc
        'target_scale': torch.tensor([center_, scale_]).unsqueeze(0), # should be the center and scale of the robust scalar
        'decoder_time_idx': torch.stack(time_idx)

    }

    '''
    output[0] = tensor of shape [2, 50, 7] # 7 because we have 7 quantiles, so 2 batches, 50 forward proj, 7 quantiles, quantile 0.5 is the prediction so it would be index 3 for the prediction
    output[1] = tensor of shape [2, 300, 1] # 2 batches, 300 for the lookback
    output[2] = tuple of length 4: 
        output[2][0]: tensor of shape [2, 300, 1] # forecasts
        output[2][1]: tensor of shape [2, 300, 1]
        output[2][2]: tensor of shape [2, 300, 1]
        output[2][3]: tensor of shape [2, 300, 1]
    output[3] = tuple of length 4:
        output[3][0]: tensor of shape [2, 50, 7] # backcasts
        output[3][1]: tensor of shape [2, 50, 7]
        output[3][2]: tensor of shape [2, 50, 7]
        output[3][3]: tensor of shape [2, 50, 7]

    '''
    
    output = projection_model(payload)
    time_idx = torch.concatenate(time_idx)

    return output, time_idx


        
def fast_gradient_attack_method(recording: pd.DataFrame, projection_model, output_path:str, epsilon = 0.5):
    adjprc = torch.from_numpy(recording["adjprc"].values)
    adjprc = adjprc.float()
    non_purturbed_features, _, _ = feature_generation(adjprc)
    columns = ["adjprc", "rolling_mean5", "rolling_mean10", "rolling_mean20", 
                             "rolling_stdev5", "rolling_stdev10", "rolling_stdev20", 
                             "log_returns", "roc5", "ema_5", "ema_10", "ema_20"]
            
    features_df = pd.DataFrame(non_purturbed_features.detach().cpu().numpy(), columns=columns)
    
    # add the specific day of the week as a feature, as stock 
    # markets tend to have a lull on Monday and improve as the week goes on
    recording["date"] = pd.to_datetime(recording["date"])
    features_df["adjprc_day"] = recording["date"].dt.day_of_week


    features_df["time_idx"] = np.arange(len(recording))

    features_df["ticker"] = recording["ticker"]

    days = torch.from_numpy(features_df["adjprc_day"].values) # we do not touch categorical features since it would be obvious

    days = days.type(torch.int)
    non_purturbed_features = non_purturbed_features.type(torch.float32)

    features_df["adjprc_day"] = features_df["adjprc_day"].astype(str)




    # Now lets do the FGSM attack (Fast Gradient Sign Method)
    normal_adjprc = adjprc.requires_grad_()
    normal_adjprc = normal_adjprc.float()
    normal_outputs, normal_time_idx = call_model(projection_model, normal_adjprc, days)

    predictions = normal_outputs[0][:, :, 3].flatten()

    normal_time_idx = normal_time_idx - 300
    max_time = max(normal_time_idx) + 1 # -300 so we start at 0
    bin_sums = torch.zeros(max_time).scatter_add(dim=0, index=normal_time_idx, src=predictions)
    bin_counts = torch.zeros(max_time).scatter_add(dim=0, index=normal_time_idx, src=torch.ones_like(predictions))

    normal_predictions = bin_sums / bin_counts
    loss = nn.functional.l1_loss(normal_predictions, adjprc[300:])
    loss = loss.float()

    projection_model.zero_grad()

    loss.backward()

    grad = adjprc.grad.data

    sign_grad = grad.sign()

    attack_adjprc = adjprc + epsilon * sign_grad


    # Now send the model the attack adjprc
    attack_outputs, attack_time_idx = call_model(projection_model, adjprc=attack_adjprc, days=days)
    predictions = attack_outputs[0][:, :, 3].flatten()
    attack_time_idx = attack_time_idx - 300
    max_time = max(attack_time_idx) + 1 # -300 so we start at 0
    bin_sums = torch.zeros(max_time).scatter_add(dim=0, index=attack_time_idx, src=predictions)
    bin_counts = torch.zeros(max_time).scatter_add(dim=0, index=attack_time_idx, src=torch.ones_like(predictions))
    attack_predictions = bin_sums / bin_counts
    
    normal_time_idx += 300
    attack_time_idx += 300

    mae = round(mean_absolute_error(adjprc[300:].detach().numpy(), normal_predictions.detach().numpy()), 3)
    mae_attack = round(mean_absolute_error(adjprc[300:].detach().numpy(), attack_predictions.detach().numpy()), 3)
    fig = plt.figure(figsize=(14, 6))
    plt.plot(adjprc.detach().numpy(), label="Actual adjprc", color="black", alpha=0.8, linestyle='--')
    plt.plot(normal_time_idx.unique(), normal_predictions.detach().numpy(), label="Normal Prediction", color="green", alpha=0.6)
    plt.plot(attack_time_idx.unique(), attack_predictions.detach().numpy(), label="Attack Prediction", color="blue", alpha=0.6)
    plt.plot(attack_adjprc.detach().numpy(), label="Attack adjprc", color="red", alpha=0.6, linestyle=":")
    plt.title(f"FGSM {features_df["ticker"].to_numpy()[0]} (all), Normal MAE: {mae}, Attack MAE: {mae_attack}")
    plt.legend()
    plt.xlabel("Day")
    plt.ylabel("adjprc")
    plt.savefig(f"{output_path}/{features_df["ticker"].to_numpy()[0]}_fgsm.png")
    plt.close()

    return mae, mae_attack


def perform_adversarial_attack(data_path, mode=0):
    model_state_dict = torch.load("NHITS_forecasting_model.pt")
    params = torch.load("./NHITS_params.pt", weights_only=False)
    params["loss"] = pf.QuantileLoss(quantiles=[0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999])
    model = pf.NHiTS(**params)
    model.load_state_dict(model_state_dict)
    total_normal_error = 0
    total_attack_error = 0
    k = 0
    output_path = "Attack_Outputs"
    initialize_directory(output_path)
    # For each recording we perform the adversarial attack
    for entry in Path(data_path).iterdir():
        if entry.suffix == ".csv":
            if mode == 0:
                df = pd.read_csv(entry)
                # here we perform the attack on the full recording

                proper_mae, attack_mae = fast_gradient_attack_method(df, model, "Attack_Outputs")

                total_normal_error += proper_mae
                total_attack_error += attack_mae
                k += 1

    total_normal_error = total_normal_error / k
    total_attack_error = total_attack_error / k

    print(f"Normal Average Error {total_normal_error}, Attack Average Error {total_attack_error}")





def plot_original_prediction(model: pf.NHiTS, test_dataloader:DataLoader, test_dataset: pf.TimeSeriesDataSet, scale_mean, scale_std, output_path:str = "./ticker_prediction.png"):
    # Make the predictions
    predictions = model.predict(test_dataloader, return_x=True, mode="prediction", trainer_kwargs=dict(accelerator="cpu"))

    # Get all our test tickers
    tickers = test_dataset.decoded_index["ticker"].unique()

    # Get the times, predictions and ground truth values
    time_idx = predictions.x["decoder_time_idx"].flatten().numpy()
    actuals = predictions.x["decoder_target"].flatten().numpy()
    preds = predictions.output.flatten().numpy()


    ticker_index = 0
    total_mae = 0
    df = pd.DataFrame({
        "time": time_idx,
        "Actual": actuals,
        "Prediction": preds
    })
    # Group by time and find the average
    df = df.groupby("time").agg({
        "Prediction" : 'mean',
        'Actual': 'mean'
    })
    # Calculate the error
    mase = mean_absolute_error(df["Actual"], df["Prediction"])
    total_mae += mase
    fig = plt.figure(figsize=(14, 6))
    plt.plot(df.index, df["Actual"], label="Actual", color="blue")
    plt.plot(df.index, df["Prediction"], label="Predicted", color="orange")
    plt.title(f"Predicted vs Actual for stock {tickers[0]} (all), MAE: {mase}")
    plt.legend()
    plt.xlabel("Day")
    plt.ylabel("adjprc")
    plt.savefig(output_path)
    plt.close()

    return df

if __name__ == '__main__':
    t1 = datetime.now()
    print(f"Started job at {t1}")

    perform_adversarial_attack("SP500_AttackData_Full")
    t2 = datetime.now()
    print(f"Finished job at {t2} with job duration {t2 - t1}")