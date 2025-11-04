from torch import nn
from h1_functions import *
from h2_functions import *

class E2ENet(nn.Module):
    def __init__(self, seq_length, n_features, y_dim,h1_mode=2,h2_constrain=1, L=1,K=0):
        super(E2ENet, self).__init__()
        self.L = L
        self.K = K
        self.y_dim = y_dim
        self.n_features = n_features
        self.h1_mode = h1_mode
        self.h2_constrain = h2_constrain
        self.model_h1 = LSTM(seq_length, n_features, y_dim)
        self.model_h2 = h2()

        if self.h1_mode == 0: self.model_h1 = LM(seq_length, n_features, y_dim)
        elif self.h1_mode == 1:self.model_h1 = MLP(seq_length, n_features, y_dim)
        elif self.h1_mode == 3: self.model_h1 = CNN(seq_length, n_features, y_dim)
        else :
            self.model_h1 = LSTM(seq_length, n_features, y_dim)
            print("wrong h1 mode, using default h1:LSTM")


    def forward(self, x):
        s=self.model_h1(x);

        if self.h2_constrain == 0: w=self.model_h2.constrain_0(s)
        if self.h2_constrain == 2: w = self.model_h2.constrain_2(s)
        if self.h2_constrain == 3: w = self.model_h2.constrain_3(s)
        if self.h2_constrain == 4: w = self.model_h2.constrain_4(s)
        else: w = self.model_h2.constrain_1(s)


        return w

