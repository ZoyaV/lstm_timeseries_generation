# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 02:57:53 2020

@author: zoya
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
import torch


class TempNN(nn.Module):
    def __init__(self, n_features, n_hidden, seq_len, n_layers=2):
        super(TempNN, self).__init__()
        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.lstm = nn.LSTM(
          input_size=n_features,
          hidden_size=n_hidden,
        )
        self.linear = nn.Linear(in_features=n_hidden, out_features=1)
        self.activate = nn.Tanh()
    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.n_layers, self.seq_len, self.n_hidden),
            torch.zeros(self.n_layers, self.seq_len, self.n_hidden)
        )
    def forward(self, sequences):
        lstm_out, self.hidden = self.lstm(
          sequences.view(len(sequences), self.seq_len, -1),
          self.hidden
        )
        last_time_step = lstm_out.view(self.seq_len, len(sequences), self.n_hidden)[-1]
        y_pred = self.linear(last_time_step)
        return y_pred

def train_model(model, X_train, y_train, X_test, y_test ):
    loss_fn = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 100
    train_hist = np.zeros(num_epochs)
    test_hist = np.zeros(num_epochs)
    model.reset_hidden_state() 
    for t in range(num_epochs):        
        print(t)
        y_pred = model(X_train)
        loss = loss_fn(y_pred.float(), y_train)        
        if X_test is not None:
            with torch.no_grad():
                y_test_pred = model(X_test)
                test_loss = loss_fn(y_test_pred.float(), y_test)
            test_hist[t] = test_loss.item()
        if t % 2 == 0:
            print(f'Epoch {t} train loss: {loss.item()} test loss: {test_loss.item()}')
        elif t % 2 == 0:
            print(f'Epoch {t} train loss: {loss.item()}')
        train_hist[t] = loss.item()
        if t!= 0 and train_hist[t-1]<train_hist[t]:
            break
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()        
    return model.eval(), train_hist, test_hist