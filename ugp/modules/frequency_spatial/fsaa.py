import torch
import torch.nn as nn
import torch.nn.functional as F

from .gft import ChebyshevFilter, get_laplacian

class FrequencySpatialAdaptiveAttention(nn.Module):
    def __init__(self, in_channels, K=3, use_gating=True):
        super().__init__()
        self.in_channels = in_channels
        self.K = K
        self.use_gating = use_gating
        
        # Spatial branch
        self.spatial_proj = nn.Linear(in_channels, in_channels)
        
        # Frequency branches (two independent Chebyshev filters to learn low/high pass automatically)
        self.low_pass = ChebyshevFilter(K, in_channels, in_channels)
        self.high_pass = ChebyshevFilter(K, in_channels, in_channels)
        
        if self.use_gating:
            # Adaptive Attention (Gating Mechanism)
            self.gating_mlp = nn.Sequential(
                nn.Linear(in_channels * 3, in_channels),
                nn.LayerNorm(in_channels), # Added LayerNorm for stability
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),           # Added Dropout to prevent overfitting
                nn.Linear(in_channels, 3)  # Output 3 weights for each point
            )
            
            # Initialize gating weights to favor spatial features initially
            nn.init.constant_(self.gating_mlp[-1].bias, 0)
            self.gating_mlp[-1].bias.data[0] = 2.0  # spatial weight alpha
            self.gating_mlp[-1].bias.data[1] = 0.0  # low-pass weight beta
            self.gating_mlp[-1].bias.data[2] = 0.0  # high-pass weight gamma
        
        # Output projection with LayerNorm
        self.out_proj = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LayerNorm(in_channels)
        )
        
        # Residual scaling factor (learnable) to prevent feature collapse
        self.gamma_res = nn.Parameter(torch.zeros(1))
        
    def forward(self, points, feats):
        """
        points: (B, N, 3) Coarse point coordinates
        feats: (B, N, C) Features to be refined
        """
        # 1. Graph Construction (Laplacian)
        L = get_laplacian(points, k=16) # (B, N, N)
        
        # 2. Extract Multi-Domain Features
        f_spatial = self.spatial_proj(feats) # (B, N, C)
        f_low = self.low_pass(L, feats)      # (B, N, C)
        f_high = self.high_pass(L, feats)    # (B, N, C)
        
        # 3. Feature Fusion
        if self.use_gating:
            # Adaptive Attention Gating
            f_concat = torch.cat([f_spatial, f_low, f_high], dim=-1) # (B, N, 3C)
            gate_weights = self.gating_mlp(f_concat) # (B, N, 3)
            gate_weights = F.softmax(gate_weights, dim=-1) # (B, N, 3)
            
            alpha = gate_weights[:, :, 0:1] # (B, N, 1)
            beta = gate_weights[:, :, 1:2]  # (B, N, 1)
            gamma = gate_weights[:, :, 2:3] # (B, N, 1)
            
            f_fused = alpha * f_spatial + beta * f_low + gamma * f_high
        else:
            # Baseline for Ablation: Simple Averaging (to keep feature scale consistent)
            f_fused = (f_spatial + f_low + f_high) / 3.0
        
        # 4. Residual Connection with Learnable Scaling
        f_fused = feats + self.gamma_res * self.out_proj(f_fused)
        
        return f_fused
