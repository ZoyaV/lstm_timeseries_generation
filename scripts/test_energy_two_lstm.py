# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 01:17:54 2020

@author: mi
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from torch import nn, optim
import torch

from model import TempNN, train_model
from preprocessing import (split_data, moving_average, scaler,
                            create_sequences, train_test, generated)


def predict_summation_batch(data, lstm, slice_size = 30):    
    """
    data - time series for forecasting.
    lstm - model for predicting trend.
    lstm2 - model for predictin diff.
    sclice_size - count of data to prediction
    """
    series_lstm1 =[]
    differs = []
    series_lstm2 = []
    for i in range(slice_size, len(data)):
        numval = np.asarray(data[i-slice_size:i]).reshape(1,slice_size)
        sc, numval = scaler(numval.ravel())
        val = torch.from_numpy(numval[:].reshape(1,-1)).float()
        predict_1 = lstm(val) 
        predict_1 = sc.inverse_transform(predict_1.detach().numpy())
       # print(predict_1)
        series_lstm1.append(predict_1)
        differs.append(data[i] - predict_1) 
        if len(series_lstm1) >= slice_size:
            lenght = len(series_lstm1)
            value = np.asarray(differs[lenght-slice_size:lenght])
            sc, numval = scaler(value.ravel())
            numval = sc.transform(np.asarray(numval[:].reshape(1,slice_size)))
            val = torch.from_numpy(numval).float()
            predict_2 = lstm(val[:1]) 
            predict_2 = sc.inverse_transform(predict_2.detach().numpy())
            series_lstm2.append(predict_2)
    series_lstm2 = np.asarray(series_lstm2)
    series_lstm1 = np.asarray(series_lstm1)
    differs = np.asarray(differs)
  #  series_lstm1 = scaler.inverse_transform(np.asarray(series_lstm1).reshape(-1,1))
    return series_lstm1[:len(series_lstm2)].ravel(), series_lstm2.ravel(), differs.ravel()

if __name__ == "__main__":
     
    values_dt = pd.read_csv('../temp_ds/Power-Networks-LCL-June2015(withAcornGps)v2_2.csv', delimiter=',')
    values_dt = np.asarray(values_dt['KWH/hh (per half hour) '].dropna(how='any',axis=0))
    values_dt[np.where(values_dt== 'Null')]=-1
    values_dt = values_dt.astype(np.float32)
    
    model = TempNN( n_features=1, n_hidden= 64, seq_len=30, n_layers=1)
    model.reset_hidden_state()
    model.load_state_dict(torch.load("../models/energy_model.pth"))
    model.eval()
    
    
    values_rm = moving_average(values_dt,20)
    batch = values_rm[70000:95000]
    batch_orig = values_dt[70000:95000]
       
    predicted = predict_summation_batch(batch_orig, model, slice_size = 30)
    summation = np.asarray(predicted[0]).ravel() + np.asarray(predicted[1]).ravel()
     
    plt.figure( figsize = (10,5))
    plt.subplot(3,1,1)
    plt.plot(np.asarray(batch_orig)[:len(batch_orig) - 30], label = "original")
    plt.legend()
    plt.subplot(3,1,2)
    plt.plot(predicted[0].ravel(), label = "lstm - 1")
    plt.legend()
    plt.subplot(3,1,3)
    plt.plot(np.asarray(batch_orig)[:len(batch_orig) - 30], label = "original")
    plt.plot(predicted[0].ravel(), label = "lstm - 1")
    plt.legend()
    mse = mean_squared_error(predicted[0], np.asarray(batch_orig)[:len(batch_orig) - 59])
    plt.title("RMSE: %.3f"% mse**0.5)
    plt.tight_layout()
    plt.show()

    plt.figure( figsize = (10,5))
    plt.subplot(3,1,1)
    plt.plot(predicted[2].ravel(), label = "difference (original - lstm_1)")
    plt.legend()
    plt.subplot(3,1,2)
    plt.plot(predicted[1].ravel(), label = "lstm - 2")
    plt.legend()
    plt.subplot(3,1,3)    
    mse = mean_squared_error(predicted[2].ravel()[:len(predicted[1])], predicted[1].ravel())
    plt.title("RMSE: %.3f"% mse**0.5)
    plt.plot(predicted[2].ravel(), label = "difference")
    plt.plot(predicted[1].ravel(), label = "lstm - 2")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure( figsize = (10,5))
    plt.subplot(3,1,1)
    plt.plot(np.asarray(batch_orig)[:len(batch_orig) - 30], label = "original")
    plt.legend()
    plt.subplot(3,1,2)
    plt.plot(summation.ravel(), label = "lstm_1 + lstm_2")
    plt.legend()
    plt.subplot(3,1,3)
    mse = mean_squared_error(np.asarray(batch_orig)[:len(summation)], summation.ravel())
    plt.title("RMSE: %.3f"% mse**0.5)
    plt.plot(np.asarray(batch_orig)[:len(batch_orig) - 30], label = "original")
    plt.plot(summation.ravel(), label = "lstm_1 + lstm_2")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    difference = np.asarray(batch_orig)[:len(summation)] - np.asarray(summation).ravel()
    with_noise = generated(difference, np.asarray(summation).ravel())
    plt.figure(figsize = (20,4)) 
    plt.plot(np.asarray(batch_orig)[:len(with_noise)], label = "original data")    
    plt.plot(np.asarray(with_noise), label = "synthetic data")
    plt.legend()
    plt.title("Generated data")
    plt.show()
    
    



