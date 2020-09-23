# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 03:01:47 2020

@author: mi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
import torch

def split_data(dt, splits_count = 20):
    all_arrays = []
    for i in range(0,len(dt)-len(dt)//splits_count,len(dt)//splits_count):
        newarr = dt[i:i + (len(dt)//splits_count)]
        all_arrays.append(newarr)
    return all_arrays
        
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def scaler(arr):
    scaler_orig = MinMaxScaler()
    scaler_orig = scaler_orig.fit(np.expand_dims(arr.ravel(), axis=1))
    data = scaler_orig.transform(np.expand_dims(arr, axis=1)) 
    return [scaler_orig, data]

def create_sequences(data, seq_length, t = 1):
    xs = []
    ys = []
    for i in range(len(data)-seq_length-t):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length:i+seq_length+t]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_test(values):
    
    test_data_size = int(len(values)*0.4)
    train_data = values[:-test_data_size]
    test_data = values[-test_data_size:]
    
    seq_length = 30
    X_train, y_train = create_sequences(train_data, seq_length, 1)
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train, y_train = X_train[indices], y_train[indices]

    X_test, y_test = create_sequences(test_data, seq_length, 1)
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()

    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()
    
    return X_train, y_train, X_test, y_test

def generated(difference, predicted):
    mu, std = norm.fit(difference)
    s = abs(np.random.normal(mu+2.5, std**0.00001, len(predicted)))
    return predicted * s 