"""
pointnet_utils.py
Tập hợp các hàm tiện ích và dataset class dùng trong notebook
pointnet_classification.ipynb.

Gồm:
    - normalize          : normalize point cloud về unit sphere
    - augment            : augmentation ngẫu nhiên cho training
    - preprocess         : tiền xử lý deterministic cho validation/inference
    - plot_cloud         : vẽ point cloud 3D bằng matplotlib
    - PointCloudDataset  : PyTorch Dataset sinh augmented sample từ file .npy
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from pathlib import Path


# Số điểm mặc định khi sample
DEFAULT_NUM_POINTS = 1024


# ── Preprocessing ─────────────────────────────────────────────────────────


def normalize(pts: np.ndarray) -> np.ndarray:
    """
    Normalize point cloud về unit sphere.

    Bước 1: trừ centroid để dịch vật thể về gốc tọa độ.
    Bước 2: chia cho max distance để scale về hình cầu đơn vị bán kính 1.

    Args:
        pts: numpy array (N, 3)

    Returns:
        numpy array (N, 3), centroid tại gốc, max distance = 1
    """
    pts = pts - pts.mean(axis=0)
    max_dist = np.max(np.linalg.norm(pts, axis=1))
    if max_dist > 0:
        pts = pts / max_dist
    return pts


def augment(pts: np.ndarray, num_points: int = DEFAULT_NUM_POINTS) -> np.ndarray:
    """
    Augment một point cloud cho training.

    Thứ tự:
        1. Xoay ngẫu nhiên theo SO(3) (cả 3 trục)
        2. Scale ngẫu nhiên trong [0.85, 1.15]
        3. Nhiễu Gaussian std = 0.02
        4. Sample num_points điểm ngẫu nhiên
        5. Normalize về unit sphere

    Args:
        pts       : numpy array (N, 3)
        num_points: số điểm sau khi sample

    Returns:
        numpy array (num_points, 3), dtype float32
    """
    # 1. SO(3) random rotation
    angles = np.random.uniform(0, 2 * np.pi, 3)
    cx, sx = np.cos(angles[0]), np.sin(angles[0])
    cy, sy = np.cos(angles[1]), np.sin(angles[1])
    cz, sz = np.cos(angles[2]), np.sin(angles[2])

    Rx = np.array([[1, 0,  0 ], [0, cx, -sx], [0,  sx, cx]])
    Ry = np.array([[cy, 0, sy], [0,  1,   0 ], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz,  0], [0,   0,  1]])
    pts = pts @ (Rz @ Ry @ Rx).T

    # 2. Scale
    pts = pts * np.random.uniform(0.85, 1.15)

    # 3. Nhiễu Gaussian
    pts = pts + np.random.randn(*pts.shape).astype(np.float32) * 0.02

    # 4. Sample
    N   = pts.shape[0]
    idx = np.random.choice(N, num_points, replace=(N < num_points))
    pts = pts[idx]

    # 5. Normalize
    return normalize(pts).astype(np.float32)


def preprocess(pts: np.ndarray, num_points: int = DEFAULT_NUM_POINTS) -> np.ndarray:
    """
    Tiền xử lý deterministic cho validation và inference.
    Không xoay hay thêm nhiễu, chỉ sample và normalize.

    Args:
        pts       : numpy array (N, 3)
        num_points: số điểm sau khi sample

    Returns:
        numpy array (num_points, 3), dtype float32
    """
    N   = pts.shape[0]
    idx = np.random.choice(N, num_points, replace=(N < num_points))
    pts = pts[idx]
    return normalize(pts).astype(np.float32)


# ── Visualization ─────────────────────────────────────────────────────────


def plot_cloud(pts: np.ndarray, ax, title: str = '',
               color: str = 'steelblue', s: float = 0.4) -> None:
    """
    Vẽ point cloud lên một Axes3D.

    Args:
        pts   : numpy array (N, 3)
        ax    : matplotlib Axes3D
        title : tiêu đề subplot
        color : màu điểm
        s     : kích thước điểm (scatter marker size)
    """
    ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1],
               s=s, c=color, alpha=0.6, linewidths=0)
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_xlabel('x', fontsize=8)
    ax.set_ylabel('z', fontsize=8)
    ax.set_zlabel('y', fontsize=8)
    ax.tick_params(labelsize=6)
    ax.set_box_aspect([1, 1, 1])
    ax.grid(False)


# ── Dataset ───────────────────────────────────────────────────────────────


class PointCloudDataset(Dataset):
    """
    Dataset sinh augmented sample từ các file .npy mẫu.

    Mỗi class có một file gốc khoảng 2048 điểm. Mỗi lần __getitem__ được
    gọi, ta lấy file gốc đó và áp augmentation ngẫu nhiên, tạo ra một
    sample mới. samples_per_class điều khiển số lần lặp (epoch size).

    Args:
        sample_dir        : thư mục chứa các file {class}_sample.npy
        classes           : danh sách tên class theo thứ tự label
        samples_per_class : số sample sinh ra cho mỗi class
        num_points        : số điểm sau khi sample
        training          : True thì augment, False thì preprocess deterministic
    """

    def __init__(self,
                 sample_dir: Path,
                 classes: list,
                 samples_per_class: int = 200,
                 num_points: int = DEFAULT_NUM_POINTS,
                 training: bool = True):

        self.num_points = num_points
        self.training   = training
        self.items      = []  # list of (raw_pts, label)

        for label, cls in enumerate(classes):
            path = Path(sample_dir) / f'{cls}_sample.npy'
            pts  = np.load(path).astype(np.float32)
            for _ in range(samples_per_class):
                self.items.append((pts, label))

        split = 'train' if training else 'val'
        print(f"[{split}] {len(self.items)} samples "
              f"({samples_per_class} per class, {len(classes)} classes)")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        pts, label = self.items[idx]
        if self.training:
            pts = augment(pts, self.num_points)
        else:
            pts = preprocess(pts, self.num_points)
        return torch.from_numpy(pts), label
