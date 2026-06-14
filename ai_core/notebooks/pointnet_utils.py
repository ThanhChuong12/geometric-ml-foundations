import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from pathlib import Path

DEFAULT_NUM_POINTS = 1024

# Preprocessing
def normalize(pts: np.ndarray) -> np.ndarray:
    pts = pts - pts.mean(axis=0)
    max_dist = np.max(np.linalg.norm(pts, axis=1))
    if max_dist > 0:
        pts = pts / max_dist
    return pts


def augment(pts: np.ndarray, num_points: int = DEFAULT_NUM_POINTS) -> np.ndarray:
    # SO(3) random rotation
    angles = np.random.uniform(0, 2 * np.pi, 3)
    cx, sx = np.cos(angles[0]), np.sin(angles[0])
    cy, sy = np.cos(angles[1]), np.sin(angles[1])
    cz, sz = np.cos(angles[2]), np.sin(angles[2])

    Rx = np.array([[1, 0,  0 ], [0, cx, -sx], [0,  sx, cx]])
    Ry = np.array([[cy, 0, sy], [0,  1,   0 ], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz,  0], [0,   0,  1]])
    pts = pts @ (Rz @ Ry @ Rx).T

    # Scale
    pts = pts * np.random.uniform(0.85, 1.15)

    # Gaussian noise
    pts = pts + np.random.randn(*pts.shape).astype(np.float32) * 0.02

    # Sample
    N   = pts.shape[0]
    idx = np.random.choice(N, num_points, replace=(N < num_points))
    pts = pts[idx]

    # Normalize
    return normalize(pts).astype(np.float32)


def preprocess(pts: np.ndarray, num_points: int = DEFAULT_NUM_POINTS) -> np.ndarray:
    N   = pts.shape[0]
    idx = np.random.choice(N, num_points, replace=(N < num_points))
    pts = pts[idx]
    return normalize(pts).astype(np.float32)


# Visualization
def plot_cloud(pts: np.ndarray, ax, title: str = '',
               color: str = 'steelblue', s: float = 0.4) -> None:
    ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1],
               s=s, c=color, alpha=0.6, linewidths=0)
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_xlabel('x', fontsize=8)
    ax.set_ylabel('z', fontsize=8)
    ax.set_zlabel('y', fontsize=8)
    ax.tick_params(labelsize=6)
    ax.set_box_aspect([1, 1, 1])
    ax.grid(False)


# Dataset
class PointCloudDataset(Dataset):
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
