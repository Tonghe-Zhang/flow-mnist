import torch
import torch.nn as nn
from typing import List
import math
import torch.nn.functional as F
from script.helpers import *


from model.mlp import MLP
import operator
from functools import reduce

class FlowMLP(nn.Module):
    def __init__(self, flow_mlp_cfg):
        super(FlowMLP, self).__init__()
        
        self.flow_mlp_cfg = flow_mlp_cfg
        
        self.data_hidden_dim = self.flow_mlp_cfg.data_hidden_dim
        
        self.time_hidden_dim = self.flow_mlp_cfg.time_hidden_dim
        
        self.cls_hidden_dim = self.flow_mlp_cfg.cls_hidden_dim

        # visual encoder
        self.encoder = MNISTEncoder(latent_dim=self.data_hidden_dim)
        
        self.time_embedder = SinusoidalTimeEmbedder(embed_dim=self.time_hidden_dim, max_steps=64)
        
        self.label_embedder = LabelEmbedder(num_classes=10, embed_dim=self.cls_hidden_dim)
        
        self.net=nn.Sequential(
            MLP(input_dim=self.data_hidden_dim + self.time_hidden_dim + self.cls_hidden_dim,
                     hidden_dims=self.flow_mlp_cfg.hidden_dims,
                     output_dim=self.flow_mlp_cfg.output_dim,
                     batch_norm=self.flow_mlp_cfg.batch_norm),
            nn.Tanh()
        )
        # we add an activation function Tanh to normalize output to [-1,1], and then scale to [0,255]. 
        
        # self.time_embedding=nn.Sequential(
        #     SinusoidalTimeEmbedder(self.time_hidden_dim),
        #     nn.Linear(self.time_hidden_dim, self.time_hidden_dim * 2),
        #     nn.Mish(),
        #     nn.Linear(self.time_hidden_dim * 2, self.time_hidden_dim),
        # )
    
    def forward(self, t, x, cls):
        '''
        inputs:
            t: torch.Size([N]), torch.float32
            x: torch.Size([N, C, H, W]), torch.float32
            cls: torch.Size([N]), torch.int64
        output:
            v_hat: torch.Size([N, C, H, W]), torch.float32
        '''
        
        # print(f"x.shape: {x.shape}")
        # print(f"t.shape = {t.shape}")
        
        time_latent  = self.time_embedder(t)                # [N, time_hidden_dim] 
        data_latent = self.encoder(x)                       # [N, data_hidden_dim]
        label_latent = self.label_embedder(cls)             # [N, cls_hidden_dim] 
        
        # print(f"data_latent.shape={data_latent.shape}")
        # print(f"time_latent.shape={time_latent.shape}")
        # print(f"label_latent.shape={label_latent.shape}")
        
        latent = torch.cat([data_latent, label_latent, time_latent], dim=1)
        
        v_hat = self.net(latent)
        
        return v_hat.view(x.shape)
    
        
        