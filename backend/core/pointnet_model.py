"""
PointNet Classification Model — PyTorch Implementation
Ported from official TensorFlow implementation by Qi et al., CVPR 2017.

Reference:
    Source: pointnet/models/pointnet_cls.py + pointnet_cls_basic.py + transform_nets.py
    Paper: Qi et al., "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation"

Author: Ported for geo-ml-web demo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────────────────────
# T-Net: Input Transform Network (3x3 spatial alignment)
# Reference: transform_nets.py::input_transform_net()
# ─────────────────────────────────────────────────────────────
class InputTransformNet(nn.Module):
    """
    Mini-network that predicts a 3x3 affine transformation matrix
    to align input point cloud to a canonical space.

    Architecture mirrors the TF1 T-Net:
    Conv(64) → Conv(128) → Conv(1024) → MaxPool → FC(512) → FC(256) → FC(9)
    Bias initialized as identity matrix [1,0,0,0,1,0,0,0,1].
    """
    def __init__(self):
        super().__init__()
        # Shared conv layers (equivalent to TF conv2d with kernel [1,3] then [1,1])
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)  # 3x3 = 9 values

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        # Initialize bias as identity matrix (matches TF: biases += [1,0,0,0,1,0,0,0,1])
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.copy_(torch.eye(3).flatten())

    def forward(self, x):
        # x: (B, 3, N)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, dim=2)[0]  # Global max pool -> (B, 1024)

        # FIX: Skip BN *only* during training with B=1 (zero variance -> NaN/undefined).
        # In eval() mode, BN uses running_mean/running_var regardless of batch size
        # -> safe for B=1. Old condition `if x.shape[0] > 1` was wrong because it
        # also skipped BN in eval mode, creating a train/test mismatch that caused
        # the T-Net to produce garbage transforms at inference time.
        if self.training and x.shape[0] == 1:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        else:
            x = F.relu(self.bn4(self.fc1(x)))
            x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        return x.view(-1, 3, 3)  # (B, 3, 3)


# ─────────────────────────────────────────────────────────────
# T-Net: Feature Transform Network (64x64 feature alignment)
# Reference: transform_nets.py::feature_transform_net()
# ─────────────────────────────────────────────────────────────
class FeatureTransformNet(nn.Module):
    """
    Mini-network that predicts a KxK transformation matrix for
    feature-space alignment. Orthogonal regularization is applied
    during training: L_reg = ||I - A·A^T||^2_F

    Architecture mirrors the TF1 feature T-Net with K=64.
    """
    def __init__(self, K=64):
        super().__init__()
        self.K = K

        self.conv1 = nn.Conv1d(K, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, K * K)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        # Initialize bias as identity matrix (matches TF: biases += np.eye(K).flatten())
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.copy_(torch.eye(K).flatten())

    def forward(self, x):
        # x: (B, K, N)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, dim=2)[0]  # Global max pool -> (B, 1024)

        # FIX: same as InputTransformNet — skip BN only when training with B=1
        if self.training and x.shape[0] == 1:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        else:
            x = F.relu(self.bn4(self.fc1(x)))
            x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        return x.view(-1, self.K, self.K)  # (B, K, K)


# ─────────────────────────────────────────────────────────────
# PointNet Basic — No T-Net (Baseline)
# Reference: pointnet_cls_basic.py::get_model()
# ─────────────────────────────────────────────────────────────
class PointNetBasic(nn.Module):
    """
    Baseline PointNet without any input or feature transformations.
    Serves as the comparison baseline to highlight the benefit of T-Nets.

    Pipeline:
        Input (B,N,3) → Shared MLP(64,64,64,128,1024) → MaxPool → FC(512,256,40)
    """
    def __init__(self, num_classes=40):
        super().__init__()
        # Shared MLP implemented as Conv1d (equivalent to TF conv2d [1,1])
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        # Classification head
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(p=0.3)  # keep_prob=0.7 in TF source

    def forward(self, x):
        """
        Args:
            x: (B, N, 3) — batch of point clouds
        Returns:
            logits: (B, num_classes)
            critical_indices: (B, 1024) — indices of critical points (Theorem 2)
        """
        B, N, _ = x.shape

        # Transpose to (B, 3, N) for Conv1d
        x = x.transpose(2, 1)

        # Shared MLP — each point processed independently
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))  # (B, 1024, N)

        # Symmetric function: Global max pooling
        # Save indices → these are the critical points (Theorem 2)
        global_feat, critical_indices = torch.max(x, dim=2)  # (B, 1024), (B, 1024)

        # Classification MLP
        x = F.relu(self.bn6(self.fc1(global_feat)))
        x = self.dropout(x)
        x = F.relu(self.bn7(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return x, critical_indices


# ─────────────────────────────────────────────────────────────
# PointNet Full — With Input + Feature Transform (T-Net)
# Reference: pointnet_cls.py::get_model()
# ─────────────────────────────────────────────────────────────
class PointNetFull(nn.Module):
    """
    Full PointNet with Input Transform (3x3) and Feature Transform (64x64).
    The feature transform uses orthogonal regularization during training:
        L_total = L_cls + reg_weight * ||I - A·A^T||^2_F

    Pipeline:
        Input (B,N,3)
        → InputTransformNet (3x3)
        → Shared MLP(64,64)
        → FeatureTransformNet (64x64)
        → Shared MLP(64,128,1024)
        → MaxPool → GlobalFeat (1024)
        → FC(512,256,40)
    """
    def __init__(self, num_classes=40):
        super().__init__()

        # T-Net modules
        self.input_transform = InputTransformNet()
        self.feature_transform = FeatureTransformNet(K=64)

        # First shared MLP block (before feature transform)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        # Second shared MLP block (after feature transform)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        # Classification head
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        """
        Args:
            x: (B, N, 3) — batch of point clouds
        Returns:
            logits:           (B, num_classes)
            critical_indices: (B, 1024)  — critical points (Theorem 2)
            trans_feat:       (B, 64, 64) — feature transform, for orthogonal reg loss
            trans_input:      (B, 3, 3)   — input transform,   for orthogonal reg loss
        """
        B, N, _ = x.shape

        # ── Input Transform (3x3) ──
        trans_input = self.input_transform(x.transpose(2, 1))   # (B, 3, 3)
        x = torch.bmm(x, trans_input)                           # (B, N, 3)

        x = x.transpose(2, 1)  # (B, 3, N) for Conv1d

        # First shared MLP
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 64, N)
        x = F.relu(self.bn2(self.conv2(x)))  # (B, 64, N)

        # ── Feature Transform (64x64) ──
        trans_feat = self.feature_transform(x)             # (B, 64, 64)
        x = x.transpose(2, 1)                             # (B, N, 64)
        x = torch.bmm(x, trans_feat)                      # (B, N, 64)
        point_feat = x.transpose(2, 1)                    # (B, 64, N) — local features

        # Second shared MLP
        x = F.relu(self.bn3(self.conv3(point_feat)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))               # (B, 1024, N)

        # Symmetric function: Global max pooling + save critical indices
        global_feat, critical_indices = torch.max(x, dim=2)  # (B, 1024), (B, 1024)

        # Classification MLP
        x = F.relu(self.bn6(self.fc1(global_feat)))
        x = self.dropout(x)
        x = F.relu(self.bn7(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        # Return 4 values: trans_input exposed so training can regularize both T-Nets
        return x, critical_indices, trans_feat, trans_input


def feature_transform_regularizer(trans):
    """
    Orthogonal regularization loss for feature transform matrix.
    L_reg = ||I - A·A^T||^2_F  (Eq. 2 in paper, reg_weight=0.001)

    Args:
        trans: (B, K, K) — feature transformation matrix
    Returns:
        Scalar loss value
    """
    K = trans.shape[1]
    I = torch.eye(K, device=trans.device).unsqueeze(0)
    diff = torch.bmm(trans, trans.transpose(2, 1)) - I
    return torch.mean(torch.norm(diff, dim=(1, 2)))
