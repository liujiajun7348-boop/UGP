import torch
import numpy as np
import os
import matplotlib
# Use headless backend for server rendering
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_laplacian(points, k=16, sigma=1.0):
    B, N, _ = points.shape
    dist_sq = torch.cdist(points, points, p=2) ** 2
    knn_val, knn_idx = torch.topk(dist_sq, k=k+1, dim=-1, largest=False)
    mask = torch.zeros_like(dist_sq).scatter_(-1, knn_idx, 1.0)
    mask = torch.max(mask, mask.transpose(1, 2))
    A = torch.exp(-dist_sq / (2 * sigma ** 2)) * mask
    A = A * (1 - torch.eye(N, device=points.device).unsqueeze(0).expand(B, -1, -1))
    D = torch.sum(A, dim=-1)
    D_inv_sqrt = torch.pow(D.clamp(min=1e-5), -0.5)
    D_inv_sqrt = torch.diag_embed(D_inv_sqrt)
    I = torch.eye(N, device=points.device).unsqueeze(0).expand(B, -1, -1)
    L = I - torch.bmm(torch.bmm(D_inv_sqrt, A), D_inv_sqrt)
    return L

def create_dummy_kitti_like_cloud():
    """Create a synthetic dataset that looks like a street scene if real data is not found"""
    ground = np.random.rand(4000, 3) * [20, 20, 0.1] - [10, 10, 0.05]
    wall = np.random.rand(2000, 3) * [0.5, 10, 5] + [8, -5, 0]
    tree = np.random.randn(500, 3) * [0.5, 0.5, 2] + [-5, 5, 2]
    car = np.random.rand(1000, 3) * [4, 2, 1.5] + [-2, -2, 0.75]
    pts = np.vstack([ground, wall, tree, car])
    pts += np.random.randn(*pts.shape) * 0.02
    return pts

def render_to_image(pts, colors, out_path):
    """Render point cloud to a high-quality 2D image suitable for a paper"""
    fig = plt.figure(figsize=(10, 8), dpi=300)
    
    # 1. Top-Down View (Bird's Eye View)
    ax1 = fig.add_subplot(221)
    ax1.scatter(pts[:, 0], pts[:, 1], c=colors, s=0.5, alpha=0.8)
    ax1.set_aspect('equal')
    ax1.set_title("Top-Down View (BEV)", fontsize=12)
    ax1.axis('off')
    
    # 2. Perspective View 1
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=0.5, alpha=0.8)
    ax2.view_init(elev=30, azim=45)
    ax2.set_title("Perspective View 1", fontsize=12)
    ax2.axis('off')
    
    # 3. Perspective View 2
    ax3 = fig.add_subplot(223, projection='3d')
    ax3.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=0.5, alpha=0.8)
    ax3.view_init(elev=10, azim=135)
    ax3.set_title("Perspective View 2", fontsize=12)
    ax3.axis('off')
    
    # 4. Colorbar Legend
    ax4 = fig.add_subplot(224)
    ax4.axis('off')
    sm = plt.cm.ScalarMappable(cmap='turbo', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax4, orientation='horizontal', fraction=0.8)
    cbar.set_label('Frequency Energy (Blue=Low/Topology, Red=High/Detail)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def main():
    try:
        # Try relative to the dataset root or the common paths
        possible_paths = [
            '/home/ljj/UGP/data/Kitti/downsampled/00/000000.npy',
            '/home/ljj/UGP/data/Kitti/downsampled/08/000000.npy',
            '/data/liujiajun/Kitti/downsampled/00/000000.npy'
        ]
        pts = None
        for p in possible_paths:
            if os.path.exists(p):
                pts = np.load(p)
                print(f"Loaded real KITTI data from {p}")
                break
        if pts is None:
            raise FileNotFoundError("Could not find any KITTI .npy files.")
            
        idx = np.random.choice(len(pts), min(8000, len(pts)), replace=False)
        pts = pts[idx]
    except Exception as e:
        print("Could not load real KITTI data, generating synthetic street scene.")
        pts = create_dummy_kitti_like_cloud()

    pts_tensor = torch.from_numpy(pts).unsqueeze(0).float()
    
    print("Computing Graph Laplacian...")
    L = get_laplacian(pts_tensor, k=16, sigma=1.0)
    
    high_freq_signal = torch.bmm(L, pts_tensor)
    energy = torch.norm(high_freq_signal, dim=-1).squeeze(0).numpy()
    
    # Robust normalization using percentiles to ignore extreme outliers
    p2, p98 = np.percentile(energy, 2), np.percentile(energy, 98)
    energy = np.clip(energy, p2, p98)
    energy = (energy - p2) / (p98 - p2 + 1e-8)
    
    cmap = plt.get_cmap('turbo')
    colors = cmap(energy)[:, :3]
    
    out_img_path = '/home/ljj/UGP/gft_visualization.png'
    print(f"Rendering images to {out_img_path}...")
    render_to_image(pts, colors, out_img_path)
    print(f"Image successfully saved to {out_img_path}")

if __name__ == '__main__':
    main()