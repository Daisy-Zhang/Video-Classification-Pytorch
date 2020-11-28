import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np

class RNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.LSTM(
            input_size = 1000,
            hidden_size = 64,
            num_layers = 1,
            batch_first = True
        )
        self.out = torch.nn.Linear(in_features = 64, out_features = 10)

    def forward(self,x):
        # 以下关于shape的注释只针对单向
        # output: [batch_size, time_step, hidden_size]
        # h_n: [num_layers,batch_size, hidden_size] # 虽然LSTM的batch_first为True,但是h_n/c_n的第一维还是num_layers
        # c_n: 同h_n
        output, (h_n, c_n) = self.rnn(x)
        print(output.size())
        # output_in_last_timestep=output[:,-1,:] # 也是可以的
        output_in_last_timestep = h_n[-1, :, :]
        print(output_in_last_timestep.size())
        # print(output_in_last_timestep.equal(output[:,-1,:])) #ture
        x = self.out(output_in_last_timestep)
        return x

def rnn():
    return RNN()
