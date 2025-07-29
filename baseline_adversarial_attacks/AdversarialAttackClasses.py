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
from abc import ABC, abstractmethod

from torch.utils.data import DataLoader

import os

class AdversarialAttack(ABC):
    def __init__(self, projection_model, epsilon = 0.5, lookback_length = 100, forecast_length = 20):
        super().__init__()

        self.projection_model = projection_model
        self.epsilon = epsilon
        self.lookback_length = lookback_length
        self.forecast_length = forecast_length

    @abstractmethod
    def attack(self, recording: pd.DataFrame):
        pass


    '''
    Compute the ema based on the pandas formula in their documentation (when adjust=False)
    '''
    def exponential_moving_average(self, adjprc: torch.Tensor, span: int):

        alpha = 2.0 / (span + 1.0)

        ema = torch.zeros_like(adjprc) 
        ema[0] = adjprc[0]

        # formula taken from the pandas definition
        for i in range(1, len(adjprc)):
            ema[i] = (1 - alpha) * adjprc[i-1] + alpha * adjprc[i]

        return ema
    
    def feature_generation(self, adjprc: torch.Tensor):

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

        rolling_stdev5 = torch.sqrt(torch.clip(rollingExp_mean5 - rolling_mean5**2 + 1e-06, min=0))
        rolling_stdev10 = torch.sqrt(torch.clip(rollingExp_mean10 - rolling_mean10**2 + 1e-06, min=0))
        rolling_stdev20 = torch.sqrt(torch.clip(rollingExp_mean20 - rolling_mean20**2 + 1e-06, min=0))
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
        ema_5 = self.exponential_moving_average(adjprc=adjprc, span=5)
        ema_10 = self.exponential_moving_average(adjprc=adjprc, span=10)
        ema_20 = self.exponential_moving_average(adjprc=adjprc, span=20)

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
    
    def call_model(self, adjprc: torch.Tensor, days):
        x_prime = adjprc

        features, scale_, center_ = self.feature_generation(x_prime)
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

        # if len(features) - self.lookback_length - self.forecast_length + 1 < 0:
        #     encoder_batches.append(features)
        #     encoder_categorical_batches.append(days)
        #     encoder_targets.append(x_prim)

        #     decoder_batches.append(torch.zeros((self.forecast_length, features.shape[-1])))
        #     decoder_categorical_batches.append(torch.zeros((self.forecast_length)))
        #     decoder_targets.append(x_prime[i+self.lookback_length:i + self.lookback_length + self.forecast_length])
        #     time_idx.append(torch.arange(i+self.lookback_length, i + self.lookback_length + self.forecast_length))

        for i in range(0, len(features) - self.lookback_length - self.forecast_length + 1): # need to make room for prediction values
            encoder_batches.append(features[i:i+self.lookback_length, :])
            encoder_categorical_batches.append(days[i:i+self.lookback_length])
            encoder_targets.append(x_prime[i:i+self.lookback_length])

            decoder_batches.append(features[i+self.lookback_length:i+self.lookback_length+self.forecast_length, :])
            decoder_categorical_batches.append(days[i+self.lookback_length : i+self.lookback_length+self.forecast_length])
            decoder_targets.append(x_prime[i+self.lookback_length:i + self.lookback_length + self.forecast_length])
            time_idx.append(torch.arange(i+self.lookback_length, i + self.lookback_length + self.forecast_length))


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
            'encoder_lengths': torch.tensor([self.lookback_length]), # should be length of lookback period
            'encoder_target': encoder_targets, # should be the target variable of teh lookback period (adjprc)
            'decoder_cat': decoder_categorical_batches, # should be the categorical features of the next self.forecast_length features
            'decoder_cont': decoder_batches, # should be the continous features of the next self.forecast_length features
            'decoder_lengths': torch.tensor([self.forecast_length]), # should be the length of the prediction
            'decoder_target': decoder_targets, # should be the ground truths for the next self.forecast_length adjprc
            'target_scale': torch.tensor([center_, scale_]).unsqueeze(0), # should be the center and scale of the robust scalar
            'decoder_time_idx': torch.stack(time_idx)

        }

        '''
        output[0] = tensor of shape [2, 50, 7] # 7 because we have 7 quantiles, so 2 batches, 50 forward proj, 7 quantiles, quantile 0.5 is the prediction so it would be index 3 for the prediction
        output[1] = tensor of shape [2, self.lookback_length, 1] # 2 batches, self.lookback_length for the lookback
        output[2] = tuple of length 4: 
            output[2][0]: tensor of shape [2, self.lookback_length, 1] # forecasts
            output[2][1]: tensor of shape [2, self.lookback_length, 1]
            output[2][2]: tensor of shape [2, self.lookback_length, 1]
            output[2][3]: tensor of shape [2, self.lookback_length, 1]
        output[3] = tuple of length 4:
            output[3][0]: tensor of shape [2, 50, 7] # backcasts
            output[3][1]: tensor of shape [2, 50, 7]
            output[3][2]: tensor of shape [2, 50, 7]
            output[3][3]: tensor of shape [2, 50, 7]

        '''
        
        output = self.projection_model(payload)
        time_idx = torch.concatenate(time_idx)

        return output, time_idx
    

    def get_days(self, recording: pd.DataFrame):
        recording["date"] = pd.to_datetime(recording["date"])
        recording["adjprc_day"] = recording["date"].dt.day_of_week
        days = torch.from_numpy(recording["adjprc_day"].values) # we do not touch categorical features since it would be obvious
        days = days.type(torch.int)
        return days
    
    def get_predictions(self, outputs, time_idx):
        predictions = outputs[0][:, :, 3].flatten()
        time_idx = time_idx - self.lookback_length
        max_time = max(time_idx) + 1 # -self.lookback_length so we start at 0
        bin_sums = torch.zeros(max_time).scatter_add(dim=0, index=time_idx, src=predictions)
        bin_counts = torch.zeros(max_time).scatter_add(dim=0, index=time_idx, src=torch.ones_like(predictions))
        final_predictions = bin_sums / bin_counts
        return final_predictions
    
class BasicIterativeMethod(AdversarialAttack):
    def __init__(self, projection_model, iterations, epsilon=0.5):
        super().__init__(projection_model, epsilon)

        self.iterations = iterations
        self.alpha = 1.5 * epsilon / self.iterations

    def attack(self, recording):
        adjprc = torch.from_numpy(recording["adjprc"].values)
        adjprc = adjprc.float()
        days = self.get_days(recording)


        x_i = adjprc

        for i in range(self.iterations):
            x_i = x_i.requires_grad_()
            x_i = x_i.float()
            self.projection_model.zero_grad()
            output, time_idx = self.call_model(adjprc=x_i, days=days)

            predictions = output[0][:, :, 3].flatten()

            time_idx = time_idx - self.lookback_length
            max_time = max(time_idx) + 1 # -self.lookback_length so we start at 0
            bin_sums = torch.zeros(max_time).scatter_add(dim=0, index=time_idx, src=predictions)
            bin_counts = torch.zeros(max_time).scatter_add(dim=0, index=time_idx, src=torch.ones_like(predictions))

            predictions = bin_sums / bin_counts
            loss = nn.functional.l1_loss(predictions, adjprc[self.lookback_length:])
            loss = loss.float()


            loss.backward()

            with torch.no_grad():
                grad = x_i.grad.data

                sign_grad = grad.sign()

                perturbation = torch.clip((x_i + self.alpha * sign_grad) - adjprc, -self.epsilon, self.epsilon)

                x_i = adjprc + perturbation

            x_i = x_i.detach()

        # Now send the model the attack adjprc
        attack_outputs, attack_time_idx = self.call_model(adjprc=x_i, days=days)
        attack_predictions = self.get_predictions(attack_outputs, attack_time_idx)

        normal_adjprc = adjprc.float()
        normal_outputs, normal_time_idx = self.call_model(normal_adjprc, days)
        normal_predictions = self.get_predictions(normal_outputs, normal_time_idx)
        
        normal_time_idx += self.lookback_length
        attack_time_idx += self.lookback_length

        mae = round(mean_absolute_error(adjprc[self.lookback_length:].detach().numpy(), normal_predictions.detach().numpy()), 3)
        mae_attack = round(mean_absolute_error(adjprc[self.lookback_length:].detach().numpy(), attack_predictions.detach().numpy()), 3)

        return (mae, normal_predictions), x_i,  (mae_attack, attack_predictions)
    
class FGSM(BasicIterativeMethod):
    def __init__(self, projection_model, epsilon=0.5):
        super().__init__(projection_model=projection_model, iterations=1, epsilon=epsilon)


class MI_FGSM(AdversarialAttack):
    def __init__(self, projection_model, iterations, decay, epsilon=0.5):
        super().__init__(projection_model, epsilon)

        self.iterations = iterations
        self.alpha = 1.5 * epsilon / iterations
        self.decay = decay

    def attack(self, recording):
        adjprc = torch.from_numpy(recording["adjprc"].values)
        adjprc = adjprc.float()
        days = self.get_days(recording)


        x_i = adjprc
        g = 0

        for i in range(self.iterations):
            x_i = x_i.requires_grad_()
            x_i = x_i.float()
            self.projection_model.zero_grad()
            output, time_idx = self.call_model(adjprc=x_i, days=days)

            predictions = output[0][:, :, 3].flatten()

            time_idx = time_idx - self.lookback_length
            max_time = max(time_idx) + 1 # -self.lookback_length so we start at 0
            bin_sums = torch.zeros(max_time).scatter_add(dim=0, index=time_idx, src=predictions)
            bin_counts = torch.zeros(max_time).scatter_add(dim=0, index=time_idx, src=torch.ones_like(predictions))

            predictions = bin_sums / bin_counts
            loss = nn.functional.l1_loss(predictions, adjprc[self.lookback_length:])
            loss = loss.float()
            loss.backward()

            with torch.no_grad():
                grad = x_i.grad.data

                g = self.decay * g + grad / grad.norm(p=1)

                sign_grad = g.sign()

                perturbation = torch.clip((x_i + self.alpha * sign_grad) - adjprc, -self.epsilon, self.epsilon)

                x_i = adjprc + perturbation

            x_i = x_i.detach()

        # Now send the model the attack adjprc
        attack_outputs, attack_time_idx = self.call_model(adjprc=x_i, days=days)
        attack_predictions = self.get_predictions(attack_outputs, attack_time_idx)

        normal_adjprc = adjprc.float()
        normal_outputs, normal_time_idx = self.call_model(normal_adjprc, days)
        normal_predictions = self.get_predictions(normal_outputs, normal_time_idx)
        
        normal_time_idx += self.lookback_length
        attack_time_idx += self.lookback_length

        mae = round(mean_absolute_error(adjprc[self.lookback_length:].detach().numpy(), normal_predictions.detach().numpy()), 3)
        mae_attack = round(mean_absolute_error(adjprc[self.lookback_length:].detach().numpy(), attack_predictions.detach().numpy()), 3)

        return (mae, normal_predictions), x_i, (mae_attack, attack_predictions)
    
'''
Based on Untargeted Algorithm from:

[1]Z. Shen and Y. Li, “Temporal characteristics-based adversarial attacks on time series forecasting,” 
Expert systems with applications, vol. 264, pp. 125950-, 2025, doi: 10.1016/j.eswa.2024.125950.
  
'''
class StealthyIterativeMethod(AdversarialAttack):
    def __init__(self, projection_model, iterations, epsilon=0.5):
        super().__init__(projection_model, epsilon)
        self.iterations = iterations
        self.alpha = 1.5 * epsilon / iterations


    def attack(self, recording):
        adjprc = torch.from_numpy(recording["adjprc"].values)
        adjprc = adjprc.float()
        days = self.get_days(recording)

        x_i = adjprc
        for i in range(self.iterations):
            x_i = x_i.requires_grad_()
            x_i = x_i.float()
            self.projection_model.zero_grad()
            output, time_idx = self.call_model(adjprc=x_i, days=days)

            predictions = output[0][:, :, 3].flatten()

            time_idx = time_idx - self.lookback_length
            max_time = max(time_idx) + 1 # -self.lookback_length so we start at 0
            bin_sums = torch.zeros(max_time).scatter_add(dim=0, index=time_idx, src=predictions)
            bin_counts = torch.zeros(max_time).scatter_add(dim=0, index=time_idx, src=torch.ones_like(predictions))

            predictions = bin_sums / bin_counts
            loss = nn.functional.l1_loss(predictions, adjprc[self.lookback_length:])
            loss = loss.float()


            loss.backward()

            with torch.no_grad():
                grad = x_i.grad.data

                sign_grad = grad.sign()

                noise = self.alpha * sign_grad

                attack_x_i = x_i + noise

                attack_x_i = torch.clamp(attack_x_i, min=adjprc - self.epsilon, max=adjprc + self.epsilon)

                # Now we have to ensure rationality of post attack forcasts where g_sim is cosign similarity

                attack_similarity = nn.functional.cosine_similarity(adjprc, attack_x_i, dim=0)
                upperbound_similarity = nn.functional.cosine_similarity(adjprc, adjprc + self.epsilon, dim=0)
                lowerbound_similarity = nn.functional.cosine_similarity(adjprc, adjprc - self.epsilon, dim=0)

                if attack_similarity > upperbound_similarity:
                    x_i = attack_x_i
                else:
                    x_i = adjprc + self.epsilon
                if attack_similarity > lowerbound_similarity:
                    x_i = attack_x_i
                else:
                    x_i = adjprc - self.epsilon

            x_i = x_i.detach()

        # Now send the model the attack adjprc
        attack_outputs, attack_time_idx = self.call_model(adjprc=x_i, days=days)
        attack_predictions = self.get_predictions(attack_outputs, attack_time_idx)

        normal_adjprc = adjprc.float()
        normal_outputs, normal_time_idx = self.call_model(normal_adjprc, days)
        normal_predictions = self.get_predictions(normal_outputs, normal_time_idx)
        
        normal_time_idx += self.lookback_length
        attack_time_idx += self.lookback_length

        mae = round(mean_absolute_error(adjprc[self.lookback_length:].detach().numpy(), normal_predictions.detach().numpy()), 3)
        mae_attack = round(mean_absolute_error(adjprc[self.lookback_length:].detach().numpy(), attack_predictions.detach().numpy()), 3)

        return (mae, normal_predictions), x_i, (mae_attack, attack_predictions)


'''
Based on Targeted Algorithm from: 

[1]Z. Shen and Y. Li, “Temporal characteristics-based adversarial attacks on time series forecasting,” 
Expert systems with applications, vol. 264, pp. 125950-, 2025, doi: 10.1016/j.eswa.2024.125950.
  
'''
class TargetedIterativeMethod(AdversarialAttack):
    def __init__(self, projection_model, iterations, direction, margin, epsilon=0.5):
        super().__init__(projection_model, epsilon)

        self.iterations = iterations
        self.alpha = 1.5 * epsilon / iterations
        self.direction = direction
        self.margin = margin

    def attack(self, recording):
        adjprc = torch.from_numpy(recording["adjprc"].values)
        adjprc = adjprc.float()
        days = self.get_days(recording)

        target = adjprc[self.lookback_length:] + self.direction * self.margin

        x_i = adjprc
        for i in range(self.iterations):
            x_i = x_i.requires_grad_()
            x_i = x_i.float()
            self.projection_model.zero_grad()
            output, time_idx = self.call_model(adjprc=x_i, days=days)

            predictions = output[0][:, :, 3].flatten()

            time_idx = time_idx - self.lookback_length
            max_time = max(time_idx) + 1 # -self.lookback_length so we start at 0
            bin_sums = torch.zeros(max_time).scatter_add(dim=0, index=time_idx, src=predictions)
            bin_counts = torch.zeros(max_time).scatter_add(dim=0, index=time_idx, src=torch.ones_like(predictions))

            predictions = bin_sums / bin_counts
            loss = nn.functional.l1_loss(predictions, target)
            loss = loss.float()


            loss.backward()

            with torch.no_grad():
                grad = x_i.grad.data

                sign_grad = grad.sign()

                noise = self.alpha * sign_grad

                attack_x_i = x_i - noise

                attack_x_i = torch.clamp(attack_x_i, min=adjprc - self.epsilon, max=adjprc + self.epsilon)

                # Now we have to ensure rationality of post attack forcasts where g_sim is cosign similarity

                # attack_similarity = nn.functional.cosine_similarity(adjprc, attack_x_i, dim=0)
                # upperbound_similarity = nn.functional.cosine_similarity(adjprc, adjprc + self.direction * self.epsilon, dim=0)

                # if attack_similarity > upperbound_similarity:
                #     x_i = attack_x_i
                # else:
                #     x_i = adjprc + self.direction * self.epsilon

                x_i = attack_x_i

            x_i = x_i.detach()

        # Now send the model the attack adjprc
        attack_outputs, attack_time_idx = self.call_model(adjprc=x_i, days=days)
        attack_predictions = self.get_predictions(attack_outputs, attack_time_idx)

        normal_adjprc = adjprc.float()
        normal_outputs, normal_time_idx = self.call_model(normal_adjprc, days)
        normal_predictions = self.get_predictions(normal_outputs, normal_time_idx)
        
        normal_time_idx += self.lookback_length
        attack_time_idx += self.lookback_length

        mae = round(mean_absolute_error(adjprc[self.lookback_length:].detach().numpy(), normal_predictions.detach().numpy()), 3)
        mae_attack = round(mean_absolute_error(adjprc[self.lookback_length:].detach().numpy(), attack_predictions.detach().numpy()), 3)

        return (mae, normal_predictions), x_i, (mae_attack, attack_predictions)
        

class Slope_Attack(AdversarialAttack):
    def __init__(self, projection_model, iterations, epsilon=0.5, target_direction = 1, c=5, d=2):
        super().__init__(projection_model, epsilon)
        self.target_direction = target_direction
        self.iterations = iterations
        self.alpha = 1.5 * epsilon/iterations
        self.c = c
        self.d = d

    def attack(self, recording):
        adjprc = torch.from_numpy(recording["adjprc"].to_numpy())
        adjprc = adjprc.float()
        days = self.get_days(recording)

        x_i = adjprc.clone()
        for i in range(self.iterations):
            x_i = x_i.requires_grad_()
            x_i = x_i.float()
            self.projection_model.zero_grad()

            output, time_idx = self.call_model(x_i, days)

            predictions = output[0][:, :, 3].flatten()

            time_idx = time_idx - self.lookback_length
            max_time = max(time_idx) + 1 # -self.lookback_length so we start at 0
            bin_sums = torch.zeros(max_time).scatter_add(dim=0, index=time_idx, src=predictions)
            bin_counts = torch.zeros(max_time).scatter_add(dim=0, index=time_idx, src=torch.ones_like(predictions))
            predictions = bin_sums / bin_counts

            slope = (predictions[-1] - predictions[0]) / len(predictions)
            if self.target_direction != 0:
                target_direction = self.target_direction * -1
                loss = self.c * torch.exp(target_direction * self.d * slope)
            else:
                loss = self.c * (slope ** 2)

            loss = loss.float()

            loss.backward()

            with torch.no_grad():
                grad = x_i.grad.data

                sign_grad = grad.sign()

                noise = self.alpha * sign_grad

                attack_x_i = x_i - noise

                attack_x_i = torch.clamp(attack_x_i, adjprc - self.epsilon, adjprc + self.epsilon)

                x_i = attack_x_i
            x_i.detach()

        # Now send the model the attack adjprc
        attack_outputs, attack_time_idx = self.call_model(adjprc=x_i, days=days)
        attack_predictions = self.get_predictions(attack_outputs, attack_time_idx)

        normal_adjprc = adjprc.float()
        normal_outputs, normal_time_idx = self.call_model(normal_adjprc, days)
        normal_predictions = self.get_predictions(normal_outputs, normal_time_idx)
        
        normal_time_idx += self.lookback_length
        attack_time_idx += self.lookback_length

        mae = round(mean_absolute_error(adjprc[self.lookback_length:].detach().numpy(), normal_predictions.detach().numpy()), 3)
        mae_attack = round(mean_absolute_error(adjprc[self.lookback_length:].detach().numpy(), attack_predictions.detach().numpy()), 3)

        return (mae, normal_predictions), x_i, (mae_attack, attack_predictions)
    
class CW_BasicSlope_Attack(AdversarialAttack):
    def __init__(self, projection_model, iterations, lr=0.01, epsilon=0.5, target_direction = 1, c=5, d=2, clamp=False):
        super().__init__(projection_model, epsilon)
        self.target_direction = target_direction
        self.iterations = iterations
        self.alpha = 1.5 * epsilon/iterations
        self.c = c
        self.d = d
        self.lr = lr
        self.clamp = clamp

    def attack(self, recording):
        adjprc = torch.from_numpy(recording["adjprc"].values)
        adjprc = adjprc.float()
        days = self.get_days(recording)


        noise = torch.zeros_like(adjprc, requires_grad=True)

        optimizer = torch.optim.Adam([noise], lr=self.lr)

        best_loss = 100000
        best_noise = 0

        
        for i in range(self.iterations):
            # x_i = x_i.requires_grad_()
            # x_i = x_i.float()
            optimizer.zero_grad()
            
            if self.clamp:
                clamped_noise = torch.clamp(noise, -self.epsilon, self.epsilon)
                x_i = adjprc + clamped_noise
            else:
                x_i = adjprc + noise
            output, time_idx = self.call_model(x_i, days)
            predictions = output[0][:, :, 3].flatten()
            #attack_predictions = self.get_predictions(attack_outputs, attack_time_idx)

            
            time_idx = time_idx - self.lookback_length
            max_time = max(time_idx) + 1 # -self.lookback_length so we start at 0
            bin_sums = torch.zeros(max_time).scatter_add(dim=0, index=time_idx, src=predictions)
            bin_counts = torch.zeros(max_time).scatter_add(dim=0, index=time_idx, src=torch.ones_like(predictions))
            predictions = bin_sums / bin_counts

            slope = (predictions[-1] - predictions[0]) / len(predictions)
            if self.target_direction != 0:
                target_direction = self.target_direction * -1
                loss = self.c * torch.exp(target_direction * self.d * slope) + torch.norm(adjprc - x_i, p=2)
            else:
                loss = self.c * (slope ** 2) + torch.norm(adjprc - x_i, p=2)
            loss = loss.float()

            if loss < best_loss:
                best_loss = loss.detach()
                best_noise = noise.detach()

            loss.backward()
            optimizer.step()

            
        #noise = torch.clamp(noise, -self.epsilon, self.epsilon)
        x_i = adjprc + best_noise

        # Now send the model the attack adjprc
        attack_outputs, attack_time_idx = self.call_model(adjprc=x_i, days=days)
        attack_predictions = self.get_predictions(attack_outputs, attack_time_idx)

        normal_adjprc = adjprc.float()
        normal_outputs, normal_time_idx = self.call_model(normal_adjprc, days)
        normal_predictions = self.get_predictions(normal_outputs, normal_time_idx)
        
        normal_time_idx += self.lookback_length
        attack_time_idx += self.lookback_length

        mae = round(mean_absolute_error(adjprc[self.lookback_length:].detach().numpy(), normal_predictions.detach().numpy()), 3)
        mae_attack = round(mean_absolute_error(adjprc[self.lookback_length:].detach().numpy(), attack_predictions.detach().numpy()), 3)

        return (mae, normal_predictions), x_i, (mae_attack, attack_predictions)
    

class LS_Slope_Attack(AdversarialAttack):
    def __init__(self, projection_model, iterations, epsilon=0.5, target_direction = 1, c=5, d=2):
        super().__init__(projection_model, epsilon)
        self.target_direction = target_direction
        self.iterations = iterations
        self.alpha = 1.5 * epsilon/iterations
        self.c = c
        self.d = d

    def attack(self, recording):
        adjprc = torch.from_numpy(recording["adjprc"].to_numpy())
        adjprc = adjprc.float()
        days = self.get_days(recording)

        x_i = adjprc.clone()
        for i in range(self.iterations):
            x_i = x_i.requires_grad_()
            x_i = x_i.float()
            self.projection_model.zero_grad()

            output, time_idx = self.call_model(x_i, days)

            predictions = output[0][:, :, 3].flatten()

            time_idx = time_idx - self.lookback_length
            max_time = max(time_idx) + 1 # -self.lookback_length so we start at 0
            bin_sums = torch.zeros(max_time).scatter_add(dim=0, index=time_idx, src=predictions)
            bin_counts = torch.zeros(max_time).scatter_add(dim=0, index=time_idx, src=torch.ones_like(predictions))
            predictions = bin_sums / bin_counts

            x = torch.arange(len(predictions), dtype=torch.float32).expand(1, len(predictions))
            x_mean = torch.mean(x).unsqueeze(-1)
            y_mean = torch.mean(predictions).unsqueeze(-1)

            numerator = ((x - x_mean) * (predictions - y_mean)).sum(dim=1)
            denom = ((x - x_mean)**2).sum(dim=1)

            slope = numerator / denom
            if self.target_direction != 0:
                target_direction = self.target_direction * -1
                loss = self.c * torch.exp(target_direction * self.d * slope)
            else:
                loss = self.c * (slope ** 2)

            loss = loss.float()

            loss.backward()

            with torch.no_grad():
                grad = x_i.grad.data

                sign_grad = grad.sign()

                noise = self.alpha * sign_grad

                attack_x_i = x_i - noise

                attack_x_i = torch.clamp(attack_x_i, adjprc - self.epsilon, adjprc + self.epsilon)

                x_i = attack_x_i
            x_i.detach()

        # Now send the model the attack adjprc
        attack_outputs, attack_time_idx = self.call_model(adjprc=x_i, days=days)
        attack_predictions = self.get_predictions(attack_outputs, attack_time_idx)

        normal_adjprc = adjprc.float()
        normal_outputs, normal_time_idx = self.call_model(normal_adjprc, days)
        normal_predictions = self.get_predictions(normal_outputs, normal_time_idx)
        
        normal_time_idx += self.lookback_length
        attack_time_idx += self.lookback_length

        mae = round(mean_absolute_error(adjprc[self.lookback_length:].detach().numpy(), normal_predictions.detach().numpy()), 3)
        mae_attack = round(mean_absolute_error(adjprc[self.lookback_length:].detach().numpy(), attack_predictions.detach().numpy()), 3)

        return (mae, normal_predictions), x_i, (mae_attack, attack_predictions)


class CW_LS_Attack(AdversarialAttack):
    def __init__(self, projection_model, iterations, lr=0.1, epsilon=0.5, target_direction = 1, c=5, d=2, clamp = False):
        super().__init__(projection_model, epsilon)
        self.target_direction = target_direction
        self.iterations = iterations
        self.alpha = 1.5 * epsilon/iterations
        self.c = c
        self.d = d
        self.lr = lr
        self.clamp = clamp

    def attack(self, recording):
        adjprc = torch.from_numpy(recording["adjprc"].values)
        adjprc = adjprc.float()
        days = self.get_days(recording)


        noise = torch.zeros_like(adjprc, requires_grad=True)

        optimizer = torch.optim.Adam([noise], lr=self.lr)

        best_loss = 100000
        best_noise = 0

        
        for i in range(self.iterations):
            # x_i = x_i.requires_grad_()
            # x_i = x_i.float()
            optimizer.zero_grad()
            
            if self.clamp:
                clamped_noise = torch.clamp(noise, -self.epsilon, self.epsilon)
                x_i = adjprc + clamped_noise
            else:
                x_i = adjprc + noise
            output, time_idx = self.call_model(x_i, days)
            predictions = output[0][:, :, 3].flatten()
            #attack_predictions = self.get_predictions(attack_outputs, attack_time_idx)

            
            time_idx = time_idx - self.lookback_length
            max_time = max(time_idx) + 1 # -self.lookback_length so we start at 0
            bin_sums = torch.zeros(max_time).scatter_add(dim=0, index=time_idx, src=predictions)
            bin_counts = torch.zeros(max_time).scatter_add(dim=0, index=time_idx, src=torch.ones_like(predictions))
            predictions = bin_sums / bin_counts

            x = torch.arange(len(predictions), dtype=torch.float32).expand(1, len(predictions))
            x_mean = torch.mean(x).unsqueeze(-1)
            y_mean = torch.mean(predictions).unsqueeze(-1)

            numerator = ((x - x_mean) * (predictions - y_mean)).sum(dim=1)
            denom = ((x - x_mean)**2).sum(dim=1)

            slope = numerator / denom
            if self.target_direction != 0:
                target_direction = self.target_direction * -1
                loss = self.c * torch.exp(target_direction * self.d * slope) + torch.norm(adjprc - x_i, p=2)
            else:
                loss = self.c * (slope ** 2) + + torch.norm(adjprc - x_i, p=2)
            loss = loss.float()

            if loss < best_loss:
                best_loss = loss.detach()
                best_noise = noise.detach()

            loss.backward()
            optimizer.step()

            
        #noise = torch.clamp(noise, -self.epsilon, self.epsilon)
        x_i = adjprc + best_noise

        # Now send the model the attack adjprc
        attack_outputs, attack_time_idx = self.call_model(adjprc=x_i, days=days)
        attack_predictions = self.get_predictions(attack_outputs, attack_time_idx)

        normal_adjprc = adjprc.float()
        normal_outputs, normal_time_idx = self.call_model(normal_adjprc, days)
        normal_predictions = self.get_predictions(normal_outputs, normal_time_idx)
        
        normal_time_idx += self.lookback_length
        attack_time_idx += self.lookback_length

        mae = round(mean_absolute_error(adjprc[self.lookback_length:].detach().numpy(), normal_predictions.detach().numpy()), 3)
        mae_attack = round(mean_absolute_error(adjprc[self.lookback_length:].detach().numpy(), attack_predictions.detach().numpy()), 3)

        return (mae, normal_predictions), x_i, (mae_attack, attack_predictions)
                
class CW_Attack(AdversarialAttack):
    def __init__(self, projection_model, iterations, c, direction, size_penalty, lr=0.01, epsilon=0.5, clamp=False):
        super().__init__(projection_model, epsilon)

        self.c = c
        self.iterations = iterations
        self.direction = direction
        self.size_penalty = size_penalty
        self.lr = lr
        self.clamp = clamp

    def attack(self, recording):
        adjprc = torch.from_numpy(recording["adjprc"].values)
        adjprc = adjprc.float()
        days = self.get_days(recording)

        if self.direction == -1:
            target = adjprc[self.lookback_length:] * 0
        if self.direction == 1:
            target = adjprc[self.lookback_length:] * 100

        noise = torch.zeros_like(adjprc, requires_grad=True)

        optimizer = torch.optim.Adam([noise], lr=self.lr)

        best_loss = 100000
        best_noise = 0

        
        for i in range(self.iterations):
            # x_i = x_i.requires_grad_()
            # x_i = x_i.float()
            optimizer.zero_grad()
            
            if self.clamp:
                clamped_noise = torch.clamp(noise, -self.epsilon, self.epsilon)
                x_i = adjprc + clamped_noise
            else:
                x_i = adjprc + noise

            attack_outputs, attack_time_idx = self.call_model(adjprc=x_i, days=days)
            attack_predictions = self.get_predictions(attack_outputs, attack_time_idx)

            
            loss = self.c * nn.functional.l1_loss(attack_predictions, target) + torch.norm(adjprc - x_i, p=2)
            loss = loss.float()

            if loss < best_loss:
                best_loss = loss.detach()
                best_noise = noise.detach()

            loss.backward()
            optimizer.step()

            
        #noise = torch.clamp(noise, -self.epsilon, self.epsilon)
        x_i = adjprc + best_noise

        # Now send the model the attack adjprc
        attack_outputs, attack_time_idx = self.call_model(adjprc=x_i, days=days)
        attack_predictions = self.get_predictions(attack_outputs, attack_time_idx)

        normal_adjprc = adjprc.float()
        normal_outputs, normal_time_idx = self.call_model(normal_adjprc, days)
        normal_predictions = self.get_predictions(normal_outputs, normal_time_idx)
        
        normal_time_idx += self.lookback_length
        attack_time_idx += self.lookback_length

        mae = round(mean_absolute_error(adjprc[self.lookback_length:].detach().numpy(), normal_predictions.detach().numpy()), 3)
        mae_attack = round(mean_absolute_error(adjprc[self.lookback_length:].detach().numpy(), attack_predictions.detach().numpy()), 3)

        return (mae, normal_predictions), x_i, (mae_attack, attack_predictions)
    


