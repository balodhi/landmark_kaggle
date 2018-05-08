import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size*2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size*2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.Softmax = nn.Softmax()
        

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)

        return out
