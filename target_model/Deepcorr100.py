import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 2000, (2,20), stride=2)
        self.max_pool1 = nn.MaxPool2d((1,5), stride=1)
        self.conv2 = nn.Conv2d(2000, 800, (4,10), stride=2)
        self.max_pool2 = nn.MaxPool2d((1,3), stride=1)
        self.fc1 = nn.Linear(9600, 3000)
        self.fc2 = nn.Linear(3000, 800)
        self.fc3 = nn.Linear(800, 100)
        self.fc4 = nn.Linear(100, 1)
#         self.d = nn.Dropout2d()
    
    def weight_init(self):
        for m in self._modules:
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.zero_()
    
    def forward(self, inp, dropout):
        x = inp
        x = F.relu(self.conv1(x))
        x = self.max_pool1(x)
        x = F.relu(self.conv2(x))
        x = self.max_pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=dropout)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=dropout)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=dropout)
        x = self.fc4(x)
        return x