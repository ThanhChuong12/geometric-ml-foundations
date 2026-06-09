"""
Train PointNet tren 5 class demo (airplane, chair, car, lamp, table).
Su dung synthetic data + augmentation de tao nhieu bien the train.

Chay lenh:
    python scripts/train_pointnet.py

Output:
    backend/models/pointnet_cls.pth   -- weights cua Full model (co T-Net)
    backend/models/pointnet_basic.pth -- weights cua Basic model (khong T-Net)

Thoi gian: ~10-20 phut tren CPU
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

# pyrefly: ignore [missing-import]
from services.pointnet_model import PointNetBasic, PointNetFull, feature_transform_regularizer

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
CLASSES       = ['airplane', 'chair', 'car', 'lamp', 'table']
NUM_CLASSES   = len(CLASSES)   # 5
NUM_POINTS    = 1024
BATCH_SIZE    = 8
EPOCHS        = 30             # 30 là đủ cho 5 class synthetic
LR            = 0.001
REG_WEIGHT    = 0.00001        # giảm 100x: tránh T-Net dominate loss gây mode collapse

DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'backend', 'data', 'sample_clouds')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'backend', 'models')

device = torch.device('cpu')


# ─────────────────────────────────────────────────────────────
# Data Augmentation
# ─────────────────────────────────────────────────────────────
def augment(pts):
    """
    Augment 1 point cloud: rotation, jitter, scale, random point drop.
    Returns new numpy array shape (NUM_POINTS, 3).
    """
    # 1. Random rotation around ALL 3 axes (SO3)
    angles = np.random.uniform(0, 2 * np.pi, 3)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]),  np.cos(angles[0])]])
    Ry = np.array([[ np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]),  np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx
    pts = pts @ R.T

    # 2. Random scale
    scale = np.random.uniform(0.85, 1.15)
    pts = pts * scale

    # 3. Random jitter (Gaussian noise)
    pts = pts + np.random.randn(*pts.shape) * 0.02

    # 4. Random point sampling
    N = pts.shape[0]
    idx = np.random.choice(N, NUM_POINTS, replace=(N < NUM_POINTS))
    pts = pts[idx]

    # 5. Normalize to unit sphere
    pts -= pts.mean(axis=0)
    d = np.max(np.linalg.norm(pts, axis=1))
    if d > 0:
        pts /= d

    return pts.astype(np.float32)


def load_raw(class_name):
    """Load raw .npy file, return (N, 3) array."""
    path = os.path.join(DATA_DIR, f"{class_name}_sample.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Sample not found: {path}\nRun: python scripts/download_assets.py")
    return np.load(path)


# ─────────────────────────────────────────────────────────────
# Dataset: generate augmented samples on-the-fly
# ─────────────────────────────────────────────────────────────
class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, samples_per_class=300, augment=True):
        self.augment = augment
        self.data    = []   # list of (pts_raw, label)

        print("Loading raw samples...")
        for label, cls in enumerate(CLASSES):
            pts = load_raw(cls)
            # Each raw sample -> samples_per_class augmented copies
            for _ in range(samples_per_class):
                self.data.append((pts, label))

        print(f"  Dataset size: {len(self.data)} samples ({samples_per_class} per class)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pts, label = self.data[idx]
        if self.augment:
            pts = augment(pts)
        else:
            # Deterministic preprocess for val
            N = pts.shape[0]
            i = np.random.choice(N, NUM_POINTS, replace=(N < NUM_POINTS))
            pts = pts[i]
            pts -= pts.mean(axis=0)
            d = np.max(np.linalg.norm(pts, axis=1))
            if d > 0:
                pts /= d
            pts = pts.astype(np.float32)
        return torch.from_numpy(pts), label


# ─────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────
def train_one_model(model, model_name, is_full, train_loader, val_loader):
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    # CosineAnnealingLR: học mượt hơn StepLR trên small dataset
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_acc  = 0.0
    save_path = os.path.join(MODELS_DIR, model_name)

    print(f"\n{'='*55}")
    print(f"  Training: {model_name}")
    print(f"  Classes: {CLASSES}")
    print(f"  Epochs: {EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LR}")
    print(f"{'='*55}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        correct    = 0
        total      = 0

        for pts, labels in train_loader:
            pts    = pts.to(device)      # (B, N, 3)
            labels = labels.to(device)

            optimizer.zero_grad()

            if is_full:
                logits, _, trans_feat = model(pts)
                cls_loss = criterion(logits, labels)
                reg_loss = feature_transform_regularizer(trans_feat)
                loss     = cls_loss + REG_WEIGHT * reg_loss
            else:
                logits, _ = model(pts)
                loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred        = logits.argmax(dim=1)
            correct    += (pred == labels).sum().item()
            total      += labels.size(0)

        scheduler.step()
        train_acc = correct / total * 100

        # Validation every 10 epochs
        if epoch % 10 == 0 or epoch == EPOCHS:
            model.eval()
            val_correct = 0
            val_total   = 0
            with torch.no_grad():
                for pts, labels in val_loader:
                    pts    = pts.to(device)
                    labels = labels.to(device)
                    if is_full:
                        logits, _, _ = model(pts)
                    else:
                        logits, _ = model(pts)
                    pred        = logits.argmax(dim=1)
                    val_correct += (pred == labels).sum().item()
                    val_total   += labels.size(0)
            val_acc = val_correct / val_total * 100

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), save_path)
                saved_marker = " <- SAVED"
            else:
                saved_marker = ""

            print(f"  Epoch {epoch:3d}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | "
                  f"Train: {train_acc:.1f}% | Val: {val_acc:.1f}%{saved_marker}")
        else:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | "
                  f"Train: {train_acc:.1f}%")

    print(f"\n  Best val accuracy: {best_acc:.1f}%")
    print(f"  Saved to: {save_path}")
    return best_acc


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Datasets
    train_set = AugmentedDataset(samples_per_class=100, augment=True)   # 100×5=500 samples
    val_set   = AugmentedDataset(samples_per_class=30,  augment=False)  # 30×5=150 samples

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # ── Train PointNet Basic (khong T-Net) ──
    basic_model = PointNetBasic(num_classes=NUM_CLASSES).to(device)
    basic_acc = train_one_model(
        basic_model, 'pointnet_basic.pth',
        is_full=False, train_loader=train_loader, val_loader=val_loader
    )

    # ── Train PointNet Full (co T-Net) ──
    full_model = PointNetFull(num_classes=NUM_CLASSES).to(device)
    full_acc = train_one_model(
        full_model, 'pointnet_cls.pth',
        is_full=True, train_loader=train_loader, val_loader=val_loader
    )

    print(f"\n{'='*55}")
    print(f"  DONE!")
    print(f"  PointNet Basic accuracy: {basic_acc:.1f}%")
    print(f"  PointNet Full  accuracy: {full_acc:.1f}%")
    print(f"")
    print(f"  Start backend:")
    print(f"  cd demo/geo-ml-web/backend && uvicorn main:app --reload")
    print(f"{'='*55}")
