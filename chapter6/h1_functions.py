import os
import sys
current_dir = os.path.dirname(os.path.abspath("__file__"))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)

import torch
import torch.nn as nn
from h2_functions import *


class LM(torch.nn.Module):
    def __init__(self, seq_length, n_features, y_dim):
        super().__init__()
        self.lm = nn.Sequential(
            nn.Flatten(),   # [512,200]
            nn.Linear(seq_length*n_features,y_dim),# [512,20]
        )

    def forward(self, x):
        s = self.lm(x)
        return s


class MLP(nn.Module):
    def __init__(self, seq_length, n_features, y_dim):
        super().__init__()#10,20,20
        self.y_dim = y_dim
        self.fc = nn.Sequential(
            nn.Flatten(),#[512,200]
            nn.Linear(seq_length*n_features, 64),#[512,4]
            nn.Tanh(),#[512,20]
            nn.Linear(64, y_dim))#[512,20]

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        s = self.fc(x)#[512,20]
        return s

class LSTM(nn.Module):
    def __init__(self, seq_length, n_features, y_dim):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=y_dim,  # 输入特征数，与卷积层输出通道数一致
            hidden_size=64,  # LSTM单元数量
            batch_first=True,  # 批处理优先
            num_layers=1
        )
        self.fc = nn.Linear(64, y_dim)

    def forward(self, x):
            #  x:(512, 10, 20)
            lstm_out, _=self.lstm(x)
            x = lstm_out[:, -1, :]#  x:(512, 10, 64)，choose the last time
            s = self.fc(x)
            return s


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
        s = self.fc(x)
        return s