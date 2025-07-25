import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.utils_model import prepare_model

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class FusionMLP_BN(nn.Module):  # ogranicz model do img - text
    def __init__(self, img_dim=512, txt_dim=768, num_labels=2, hidden_dim=256, dropout=0.3, sigma=0.0, additional_params=None):
        super().__init__()
        self.encoder1 = prepare_model(additional_params['encoder1_params'])
        self.encoder2 = prepare_model(additional_params['encoder2_params'])
        self.bn = nn.BatchNorm1d(img_dim + txt_dim)
        self.sigma = sigma  # szum do augmentacji
        self.fusion = nn.Sequential(
            nn.Linear(img_dim + txt_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, x1, x2, enable_left_branch, enable_right_branch):
        assert enable_left_branch and enable_right_branch
        x1 = self.encoder1(x1).detach()
        x2 = self.encoder2(x2).last_hidden_state[:,0,:].detach()

        # print("x2", x2.shape)  # Debugging line to check the shape of x2
        # print('x1', x1.shape)
        if not enable_left_branch:
            x1 = torch.zeros_like(x1)
        if not enable_right_branch:
            x2 = torch.zeros_like(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.bn(x)
        if self.training and self.sigma > 0.0:
            noise = torch.randn_like(x) * self.sigma
            x = x + noise
        return self.fusion(x)
    

class ResidualMLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.bn = nn.BatchNorm1d(dim)
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn(out)
        return x + out  # dodanie wejścia (residual)

class FusionMLP_EXE(nn.Module):
    def __init__(self, img_dim, txt_dim, proj_dim=256, res_hidden_dim=512, fusion_hidden_dim=256, num_labels=1, dropout=0.3, additional_params=None):
        super().__init__()
        self.encoder1 = prepare_model(additional_params['encoder1_params'])
        self.encoder2 = prepare_model(additional_params['encoder2_params'])
        # PROJEKCJE
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, proj_dim),
            nn.ReLU(),
            nn.BatchNorm1d(proj_dim)
        )
        self.txt_proj = nn.Sequential(
            nn.Linear(txt_dim, proj_dim),
            nn.ReLU(),
            nn.BatchNorm1d(proj_dim)
        )
        self.bn = nn.BatchNorm1d(proj_dim*2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # RESIDUAL MLP
        self.res_mlp = ResidualMLP(dim=proj_dim*2, hidden_dim=res_hidden_dim)
        # KOŃCOWY MLP
        self.final_mlp = nn.Sequential(
            nn.Linear(proj_dim*2, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, num_labels)
        )
    def forward(self, x1, x2, enable_left_branch, enable_right_branch):
        assert enable_left_branch and enable_right_branch
        # Enkodery
        x1 = self.encoder1(x1).detach()  # [B, img_dim]
        x2 = self.encoder2(x2).last_hidden_state[:,0,:].detach()  # [B, txt_dim]
        # Projekcje
        x1 = self.img_proj(x1)
        x2 = self.txt_proj(x2)
        # Połączenie (concat jak w CLIP-style)
        x = torch.cat([x1, x2], dim=1)
        x = self.bn(x)
        x = self.relu(x) 
        x = self.dropout(x)
        # Residual MLP
        x = self.res_mlp(x)
        # Końcowy MLP
        out = self.final_mlp(x)
        return out