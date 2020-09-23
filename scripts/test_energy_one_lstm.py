# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 03:10:33 2020

@author: zoya
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
import torch

from model import TempNN, train_model
from preprocessing import (split_data, moving_average, scaler,
                            create_sequences, train_test, generated)


def predict_batch(data, lstm, slice_size = 30):    
    """
    data - time series for forecasting.
    lstm - model for predicting trend.
    sclice_size - count of data to prediction
    """
    series_lstm1 =[]
    differs = []
    
    series_lstm2 = []
    for i in range(slice_size, len(data)):
        numval = np.asarray(data[i-slice_size:i]).reshape(1,slice_size)
        sc, numval = scaler(numval.ravel())
        
     #   numval = numval.ravel()
        val = torch.from_numpy(numval[:].reshape(1,-1)).float()
       # print(val)
        predict_1 = lstm(val) 
        predict_1 = sc.inverse_transform(predict_1.detach().numpy())
       # print(predict_1)
        series_lstm1.append(predict_1)
    return series_lstm1

if __name__ == "__main__":
    
    values_dt = pd.read_csv('../temp_ds/Power-Networks-LCL-June2015(withAcornGps)v2_2.csv', delimiter=',')
    values_dt = np.asarray(values_dt['KWH/hh (per half hour) '].dropna(how='any',axis=0))
    values_dt[np.where(values_dt== 'Null')]=-1
    values_dt = values_dt.astype(np.float32)
    
    model = TempNN( n_features=1, n_hidden= 64, seq_len=30, n_layers=1)
    model.reset_hidden_state()
    model.load_state_dict(torch.load("../models/energy_model.pth"))
    model.eval()
    
    # model = model = torch.load("../models/energy_model.pth")
    # model.eval()
    
    values_rm = moving_average(values_dt,20)
    batch = values_rm[70000:95000]
    batch_orig = values_dt[70000:95000]

    predicted = predict_batch(batch, model, slice_size = 30)
    
    plt.figure(figsize = (20,4))
    plt.subplot(1,3,1)
    plt.plot(batch)
    plt.title ("original_rolling")
    plt.subplot(1,3,2)
    plt.plot(np.asarray(predicted).ravel())
    plt.title ("predicted")
    plt.subplot(1,3,3)
    plt.plot(batch, label = "original_rolling")
    plt.plot(np.asarray(predicted).ravel(), label = "predicted")
    plt.legend()
    plt.show()
    
    plt.figure(figsize = (20,4)) 
    
    plt.subplot(2,1,1)
    plt.plot(batch, label = "original_rolling")
    plt.plot(np.asarray(predicted).ravel(), label = "predicted")
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(np.asarray(batch_orig)[:24970] - np.asarray(predicted).ravel(), label = "difference")
    plt.plot(np.asarray(predicted).ravel(), label = "predicted")
    plt.legend()
    plt.show()
    
    difference = np.asarray(batch_orig)[:len(batch)-30] - np.asarray(predicted).ravel()
    with_noise = generated(difference, np.asarray(predicted).ravel())
    plt.figure(figsize = (20,4)) 
    plt.plot(np.asarray(batch_orig)[:24970], label = "original data")
    plt.plot(with_noise, label = "synthetic data")
    plt.legend()
    plt.title("Generated data")
    plt.show()




















