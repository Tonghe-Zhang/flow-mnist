import torch
import torch.nn as nn
from typing import List

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, act=nn.ReLU(), batch_norm=False):
        super(MLP, self).__init__()
        
        # Construct an MLP net
        nets = []
        nets.append(nn.Linear(input_dim, hidden_dims[0]))
        if batch_norm:
            nets.append(nn.BatchNorm1d(hidden_dims[0]))  # Add BatchNorm after the first linear layer
        nets.append(act)

        for i in range(len(hidden_dims) - 1):
            nets.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if batch_norm:
                nets.append(nn.BatchNorm1d(hidden_dims[i + 1]))  # Add BatchNorm after each linear layer
            nets.append(act)

        nets.append(nn.Linear(hidden_dims[-1], output_dim))
        self.net = nn.Sequential(*nets)
    
    def forward(self,x):
        return self.net(x)
    
        
        