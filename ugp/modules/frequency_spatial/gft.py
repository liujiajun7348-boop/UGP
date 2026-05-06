import torch
import torch.nn as nn
import torch.nn.functional as F

def get_laplacian(points, k=16, sigma=1.0):
    """
    Constructs normalized Laplacian matrix from point cloud coordinates.
    points: (B, N, 3)
    returns: L (B, N, N) Normalized Laplacian
    """
    B, N, _ = points.shape
    k = min(k, N - 1)
    if k <= 0:
        return torch.zeros((B, N, N), device=points.device)
    
    dist_sq = torch.cdist(points, points, p=2) ** 2  # (B, N, N)
    
    # KNN mask
    knn_val, knn_idx = torch.topk(dist_sq, k=k+1, dim=-1, largest=False)
    mask = torch.zeros_like(dist_sq).scatter_(-1, knn_idx, 1.0)
    
    # Make symmetric
    mask = torch.max(mask, mask.transpose(1, 2))
    
    # Adjacency matrix (Gaussian kernel)
    A = torch.exp(-dist_sq / (2 * sigma ** 2)) * mask
    
    # Remove self loops
    A = A * (1 - torch.eye(N, device=points.device).unsqueeze(0).expand(B, -1, -1))
    
    # Degree matrix
    D = torch.sum(A, dim=-1)
    
    # Normalized Laplacian: L = I - D^(-1/2) * A * D^(-1/2)
    # Using a larger epsilon to prevent division by zero in edge cases
    D_inv_sqrt = torch.pow(D.clamp(min=1e-5), -0.5)
    D_inv_sqrt = torch.diag_embed(D_inv_sqrt)
    
    I = torch.eye(N, device=points.device).unsqueeze(0).expand(B, -1, -1)
    L = I - torch.bmm(torch.bmm(D_inv_sqrt, A), D_inv_sqrt)
    
    return L

class ChebyshevFilter(nn.Module):
    def __init__(self, K, in_channels, out_channels):
        super().__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Trainable parameters for the Chebyshev polynomials
        self.theta = nn.Parameter(torch.Tensor(K, in_channels, out_channels))
        nn.init.xavier_uniform_(self.theta)
        
    def forward(self, L, X):
        """
        L: (B, N, N) Normalized Laplacian (eigenvalues in [0, 2])
        X: (B, N, C) Features
        """
        B, N, C = X.shape
        
        # Scale Laplacian to [-1, 1].
        # Normalized Laplacian eigenvalues are in [0, 2].
        # L_scaled = L - I
        I = torch.eye(N, device=L.device).unsqueeze(0).expand(B, -1, -1)
        L_scaled = L - I
        
        # Chebyshev polynomials T_k(L_scaled)
        # T_0(L) = I -> T_0 * X = X
        # T_1(L) = L_scaled -> T_1 * X = L_scaled * X
        # T_k(L) = 2 * L_scaled * T_{k-1}(L) - T_{k-2}(L)
        
        T_k = []
        T_k.append(X) # T_0 * X
        if self.K > 1:
            T_k.append(torch.bmm(L_scaled, X)) # T_1 * X
            
        for k in range(2, self.K):
            T_k.append(2 * torch.bmm(L_scaled, T_k[-1]) - T_k[-2])
            
        # Combine with trainable weights
        out = torch.zeros(B, N, self.out_channels, device=X.device)
        for k in range(self.K):
            # T_k[k]: (B, N, C)
            # self.theta[k]: (C, out_channels)
            out += torch.matmul(T_k[k], self.theta[k])
            
        return out
