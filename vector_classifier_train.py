import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable

class CNN_(nn.Module):
    def __init__(self):
        super(CNN_, self).__init__()
        self.fc_1 = nn.Linear(12 * 1024, 1024*6)
        self.fc_2 = nn.Linear(6 * 1024, 1024)
        self.fc_3 = nn.Linear(1024, 2)
        
    def forward(self, x):
        
        out = self.fc_1(x)
        out = self.fc_2(out)
        out = self.fc_3(out)
        return out