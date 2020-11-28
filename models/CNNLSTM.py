import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np

class CNNLSTM(nn.Module):
    # Input size: 30x224x224
    # The CNN structure is first trained from single frame, then the lstm is fine-tuned from scratch.
    def __init__(self, num_class):
        super(CNNLSTM, self).__init__()

        self.lstm = nn.LSTM(1000, 512, 5, batch_first = True)
        self.fc = nn.Linear(512, num_class)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        fc_out = self.fc(lstm_out[:, -1, :])
        return fc_out

def cnnLstm(num_class):
    return CNNLSTM(num_class)