import os
import sys
current_dir = os.path.dirname(os.path.abspath("__file__"))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn

from Utilis.early_stopper import EarlyStopping
from Utilis.metrics import report_metrics
from Utilis.torch_data import MyDataset


def read_data(data_path):
    df = pd.read_csv(data_path)
    df.index = pd.to_datetime(df['Date'])
    df = df.drop(columns=['Date'])
    train_data = df[:'2017']
    val_data = pd.concat((df.loc["2018"], df.loc['2019']))
    test_data = pd.concat((df.loc["2020"], df.loc['2022':]))
    return train_data, val_data, test_data


def data_classification(X, k, T):
    [N, D] = X.shape
    df = np.array(X)
    dataX = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T:i, :]
    return dataX[:-k]


def prepare_y(data, k, T):
    return np.array(data[T - 1 + k:])

def Neg_Sharpe(portfolio):
    return -torch.mean(portfolio)/torch.std(portfolio)

class Utility(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, future_rets):
        portflio = outputs * future_rets
        portflio = torch.sum(portflio, dim=1)
        loss = Neg_Sharpe(portflio)
        return loss


class MLP(nn.Module):
    def __init__(self, seq_length, n_features, y_dim):
        super().__init__()#10,20,20
        self.fc = nn.Sequential(
            nn.Flatten(),#[512,200]
            nn.Linear(seq_length*n_features, 4),#[512,4]
            nn.Tanh(),#[512,20]
            nn.Linear(4, y_dim))#[512,20]

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)#[512,20]
        y = torch.softmax(x, dim=1) #long only+|wt|=1
        return y

class LSTM(nn.Module):
    def __init__(self, seq_length, n_features, y_dim):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=20,  # 输入特征数，与卷积层输出通道数一致
            hidden_size=64,  # LSTM单元数量
            batch_first=True,  # 批处理优先
            bidirectional=False  # 单向LSTM，如需双向可设为True
        )
        self.fc = nn.Linear(64, y_dim)

    def forward(self, x):
            #  x:(512, 10, 20)
            lstm_out, _=self.lstm(x)
            x = lstm_out[:, -1, :]#  x:(512, 10, 64)，choose the last time
            x = self.fc(x)
            y = torch.softmax(x, dim=1)  # long only+|wt|=1
            return y


class CNN(torch.nn.Module):
    def __init__(self, seq_length, n_features, y_dim):
        super().__init__()
        self.cnn_3 = torch.nn.Sequential(#[512,20，10]
            nn.Conv1d(n_features,32, kernel_size=3,padding=1),#[512,32，10]
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),#[512,64，10]
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),#[512,128，10]
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=128,  # 输入特征数，与卷积层输出通道数一致
            hidden_size=64,  # LSTM单元数量
            batch_first=True,  # 批处理优先
            bidirectional=False  # 单向LSTM，如需双向可设为True
        )
        self.fc = nn.Linear(64, y_dim)

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.cnn_3(x)       # (512, 128, 10)
        x = x.permute(0, 2, 1)  # (512, 10, 128)
        lstm_out, _ = self.lstm(x)  # (512, 10, 64)
        x = lstm_out[:, -1, :]
        x = self.fc(x)
        y = torch.softmax(x, dim=1)  # long only+|wt|=1
        return y


def train_model(model, train_data_loader, val_data_loader, X_test, savepath, epochs=5, lr=1e-3, patience=5):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = Utility()
    early_stopper = EarlyStopping(savepath=savepath, patience=patience, min_delta=1e-4, verbose=True)
    for epoch in range(epochs):
        train_epoch_loss = []
        val_epoch_loss = []

        model.train()
        for batch_X, batch_y in train_data_loader:
            optimizer.zero_grad()
            out = model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.item())

        model.eval()
        for batch_X, batch_y in val_data_loader:
            with torch.no_grad():
                out = model(batch_X)
                loss = criterion(out, batch_y)
                val_epoch_loss.append(loss.item())

        train_epoch_loss = np.mean(train_epoch_loss)
        val_epoch_loss = np.mean(val_epoch_loss)

        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train loss: {train_epoch_loss:.4f}, "
              f"Validation loss: {val_epoch_loss:.4f}")

        early_stopper(model, val_epoch_loss)
        if early_stopper.early_stop:
            print("Early stopping triggered!")
            break

    model.load_state_dict(torch.load(savepath))
    model.eval()
    y_pred_test = model(torch.from_numpy(X_test.astype(np.float32)))
    return model, y_pred_test.detach().cpu().numpy()

