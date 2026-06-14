import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import math


# SimpleCNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# PCA Logic
def get_pca_angles_batch(images):
    B = images.shape[0]
    angles = torch.zeros(B, device=images.device)
    for i in range(B):
        img = images[i, 0]
        threshold = 0.2
        y_indices, x_indices = torch.where(img > threshold)
        if len(x_indices) < 2:
            angles[i] = 0.0
            continue
            
        x_coords = x_indices.float()
        y_coords = y_indices.float()
        
        x_mean = torch.mean(x_coords)
        y_mean = torch.mean(y_coords)
        x_centered = x_coords - x_mean
        y_centered = y_coords - y_mean
        
        coords = torch.stack([x_centered, y_centered], dim=0)
        cov_matrix = torch.matmul(coords, coords.T) / (len(x_coords) - 1)
        
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        principal_vector = eigenvectors[:, 1]
        
        angle_rad = torch.atan2(principal_vector[1], principal_vector[0])
        angle_deg = math.degrees(angle_rad.item())
        angles[i] = angle_deg
    return angles


# FrameAveragingCNN
class FrameAveragingCNN(nn.Module):
    def __init__(self, backbone):
        super(FrameAveragingCNN, self).__init__()
        self.backbone = backbone
        
    def forward(self, x):
        B = x.shape[0]
        angles = get_pca_angles_batch(x)
        
        rotated_images = []
        for i in range(B):
            base_angle = -angles[i].item()
            for offset in [0, 90, 180, 270]:
                rot_img = TF.rotate(x[i], base_angle + offset)
                rotated_images.append(rot_img)
        
        rotated_batch = torch.stack(rotated_images, dim=0)
        logits = self.backbone(rotated_batch)
        logits = logits.view(B, 4, 10)
        avg_logits = torch.mean(logits, dim=1)
        return avg_logits
