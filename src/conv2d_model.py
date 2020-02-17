import numpy as np
np.random.seed(1001)

import os
import sys
import shutil

import tqdm

from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Sequential, Module
from torch.nn import Linear, Conv2d, MaxPool2d, AdaptiveMaxPool2d, Flatten, Dropout
from torch.nn import Dropout, ReLU, BatchNorm2d
from torch.nn import Softmax, LogSoftmax 

class Conv2D_Net(Module):
    
    def __init__(self):
        super(Conv2D_Net,self).__init__()
        
        self.cnn_bloc_1 = Sequential(
            ## First conv bloc
            Conv2d(in_channels=1, out_channels=32, kernel_size=(4,10)),
            ReLU(),
            BatchNorm2d(num_features=32),
            MaxPool2d(kernel_size=2)        
        )
        
        self.cnn_bloc_2 = Sequential(
            ## second conv bloc
            Conv2d(in_channels=32, out_channels=32, kernel_size=(4,10)),
            ReLU(),
            BatchNorm2d(num_features=32),
            MaxPool2d(kernel_size=2)           
        )
        
        self.cnn_bloc_3 = Sequential(
            ## Third conv bloc
            Conv2d(in_channels=32, out_channels=32, kernel_size=(4,10)),
            ReLU(),
            BatchNorm2d(num_features=32),
            MaxPool2d(kernel_size=2)  
        )
        
        self.cnn_bloc_4 = Sequential(
            ## Fourth conv bloc
            Conv2d(in_channels=32, out_channels=32, kernel_size=(4,10)),
            ReLU(),
            BatchNorm2d(num_features=32),
            MaxPool2d(kernel_size=2),
            Flatten()
        )
        
        self.linear_layers = Sequential(
            Linear(in_features = 160, out_features = 64),
            ReLU(),
#             BatchNorm1d(num_features=64),
            Linear(in_features = 64, out_features = 41),
#             BatchNorm2d(num_features=41)
#             ReLU(),
#             Linear(in_features = 1028, out_features = 41)
#             LogSoftmax(dim=1)
        )
    
    def forward(self,x):
        x = self.cnn_bloc_1(x)
        x = self.cnn_bloc_2(x)
        x = self.cnn_bloc_3(x)
        x = self.cnn_bloc_4(x)
        x = self.linear_layers(x)
#         x = F.log_softmax(x)
        return x

