import torch
import torch.nn as nn
import torch.nn.functional as F


class Decode(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(Decode, self).__init__()
        self.f1 = nn.Linear(input_dim, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.f2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.f1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.f2(x)
        return x
