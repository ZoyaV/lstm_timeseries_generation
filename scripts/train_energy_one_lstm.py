# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 03:03:29 2020

@author: mi
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


if __name__ == "__main__":   
    
    values_dt = pd.read_csv('../temp_ds/Power-Networks-LCL-June2015(withAcornGps)v2_2.csv', delimiter=',')
    values_dt = np.asarray(values_dt['KWH/hh (per half hour) '].dropna(how='any',axis=0))
    values_dt[np.where(values_dt== 'Null')]=-1
    values_dt = values_dt.astype(np.float32)
    
    splited = split_data(values_dt, 50) #Нарезаем на 50 батчей
    avg_splited = [moving_average(splited[i],20) for i in range(len(splited))] #Усредняем
    scalers_data = np.asarray([scaler(avg_splited[i]) for i in range(len(avg_splited))]) #Нормализуем
    datas = scalers_data[:,1] # Данные (батчи)
    scalers = scalers_data[:,0] # Скейлеры
    
    model = TempNN( n_features=1, n_hidden= 64, seq_len=30, n_layers=1)    
    for i,data in enumerate(datas):
        print("Batch №%d"%i)
        X_train, y_train, X_test, y_test = train_test(data)
        y_train = torch.reshape(y_train,(-1, 1))
        y_test = torch.reshape(y_test,(-1, 1)) 
        model, train_hist, test_hist = train_model( model, X_train, y_train, X_test, y_test)
    torch.save(model.state_dict(), "../models/energy_model.pth")