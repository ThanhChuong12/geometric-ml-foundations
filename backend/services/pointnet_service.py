"""
PointNet Inference Service — Backend logic cho Part 2 demo.

Chức năng:
    - Load 2 model (Basic + Full) lên memory một lần duy nhất
    - Preprocess point cloud (normalize về unit sphere)
    - Classify với cả 2 model, trả về top-3 kết quả
    - Trích xuất Critical Point Set (Theorem 2 trong paper)
    - Hỗ trợ thử nghiệm với num_points khác nhau

Không ảnh hưởng đến Part 1 (model_service.py).
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn.functional as F

from core.pointnet_model import PointNetBasic, PointNetFull

# ─────────────────────────────────────────────────────────────
# Paths & Constants
# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, '..', '..', 'storage')
WEIGHTS_DIR = os.path.join(STORAGE_DIR, 'weights')
DATA_DIR    = os.path.join(STORAGE_DIR, 'data')

CLASSES_JSON   = os.path.join(DATA_DIR, 'modelnet40_classes.json')
SAMPLE_CLOUDS  = os.path.join(DATA_DIR, 'sample_clouds')
WEIGHTS_FULL  = os.path.join(WEIGHTS_DIR, 'pointnet_cls.pth')
WEIGHTS_BASIC = os.path.join(WEIGHTS_DIR, 'pointnet_basic.pth')

NUM_CLASSES = 5   # Demo: airplane, chair, car, lamp, table
DEFAULT_NUM_POINTS = 1024

# Tự động nhận diện thiết bị (đồng bộ với Part 1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Demo class names (5 classes matching training)
_FALLBACK_CLASSES = ['airplane', 'chair', 'car', 'lamp', 'table']

# Mapping demo class -> ModelNet40 full index (for display only)
MODELNET40_INDEX = {
    'airplane': 0, 'chair': 8, 'car': 7, 'lamp': 19, 'table': 33
}

# ─────────────────────────────────────────────────────────────
# Global state — models loaded once at startup
# ─────────────────────────────────────────────────────────────
_basic_model: PointNetBasic | None = None
_full_model: PointNetFull | None  = None
_class_names: list[str] | None    = None
_models_loaded: bool               = False


def _load_class_names() -> list[str]:
    """Load demo class names (5 classes)."""
    global _class_names
    if _class_names is not None:
        return _class_names
    _class_names = _FALLBACK_CLASSES
    return _class_names


def load_models_if_needed() -> bool:
    """
    Khởi tạo và load weights cho cả 2 model.
    Chỉ chạy 1 lần duy nhất (lazy loading).

    Returns:
        True nếu ít nhất 1 model đã load thành công.
    """
    global _basic_model, _full_model, _models_loaded

    if _models_loaded:
        return True

    # ── Khởi tạo architectures ──
    _basic_model = PointNetBasic(num_classes=NUM_CLASSES).to(device)
    _full_model  = PointNetFull(num_classes=NUM_CLASSES).to(device)

    # ── Load pretrained weights ──
    if os.path.exists(WEIGHTS_FULL):
        try:
            state_dict = torch.load(WEIGHTS_FULL, map_location=device)
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            _full_model.load_state_dict(state_dict, strict=True)
            print(f"[PointNet] Loaded Full model weights: {WEIGHTS_FULL}")
        except Exception as e:
            print(f"[PointNet WARNING] Could not load Full weights: {e}")
    else:
        print(f"[PointNet WARNING] Full weights not found: {WEIGHTS_FULL}")
        print("  Run: python scripts/train_pointnet.py")

    if os.path.exists(WEIGHTS_BASIC):
        try:
            state_dict = torch.load(WEIGHTS_BASIC, map_location=device)
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            _basic_model.load_state_dict(state_dict, strict=True)
            print(f"[PointNet] Loaded Basic model weights: {WEIGHTS_BASIC}")
        except Exception as e:
            print(f"[PointNet WARNING] Could not load Basic weights: {e}")
    else:
        print(f"[PointNet WARNING] Basic weights not found: {WEIGHTS_BASIC}")

    # ── Set eval mode ──
    _basic_model.eval()
    _full_model.eval()
    _models_loaded = True
    return True


# ─────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────
def preprocess_point_cloud(points: list | np.ndarray, num_points: int = DEFAULT_NUM_POINTS) -> np.ndarray:
    """
    Normalize point cloud về unit sphere và sample đúng num_points điểm.
    Giống với cách xử lý trong train.py của source gốc.

    Steps:
        1. Convert sang numpy array (N, 3)
        2. Farthest-point sampling hoặc random sampling để lấy num_points điểm
        3. Trừ centroid → dịch về gốc tọa độ
        4. Chia cho max distance → normalize về unit sphere

    Args:
        points: list hoặc numpy array hình dạng (N, 3)
        num_points: số lượng điểm sau khi sample (mặc định 1024)

    Returns:
        numpy array shape (1, num_points, 3) — sẵn sàng đưa vào model
    """
    pts = np.array(points, dtype=np.float32)

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected shape (N, 3), got {pts.shape}")

    N = pts.shape[0]

    # ── Sampling ──
    if N >= num_points:
        # Random sample (giống train.py: uniformly sample from 2048 points)
        idx = np.random.choice(N, num_points, replace=False)
    else:
        # Nếu ít điểm hơn yêu cầu → sample lặp lại
        idx = np.random.choice(N, num_points, replace=True)

    pts = pts[idx]  # (num_points, 3)

    # ── Normalize về unit sphere ──
    centroid = pts.mean(axis=0)
    pts = pts - centroid
    max_dist = np.max(np.linalg.norm(pts, axis=1))
    if max_dist > 0:
        pts = pts / max_dist

    return pts[np.newaxis, ...]  # (1, num_points, 3)

# ─────────────────────────────────────────────────────────────
# Perturbation (thí nghiệm robustness)
# ─────────────────────────────────────────────────────────────
def apply_perturbation(
    pts: np.ndarray,
    rotation_x: float = 0.0,
    rotation_y: float = 0.0,
    rotation_z: float = 0.0,
    noise_level: float = 0.0,
    drop_ratio: float = 0.0,
) -> np.ndarray:
    """
    Áp dụng biến đổi vào point cloud (N, 3) trước khi inference.
    Dùng để kiểm chứng robustness của PointNet.

    Theo Section 5.2 của paper:
    - Rotation: kiểm tra T-Net có giúp bất biến với xoay không
    - Noise:    kiểm tra robust với nhiễu Gaussian
    - Drop:     kiểm tra Theorem 2 (critical point set)
    """
    pts = pts.copy()

    # 1. Xoay 3D (theo góc Euler)
    if rotation_x != 0 or rotation_y != 0 or rotation_z != 0:
        rx = np.radians(rotation_x)
        ry = np.radians(rotation_y)
        rz = np.radians(rotation_z)
        Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
        Ry = np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]])
        Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
        R = Rz @ Ry @ Rx
        pts = pts @ R.T

    # 2. Thêm nhiễu Gaussian
    if noise_level > 0:
        pts = pts + np.random.randn(*pts.shape).astype(np.float32) * noise_level

    # 3. Bỏ ngẫu nhiên X% điểm
    if drop_ratio > 0:
        N = pts.shape[0]
        keep = int(N * (1.0 - drop_ratio))
        keep = max(keep, 10)  # giữ ít nhất 10 điểm
        idx = np.random.choice(N, keep, replace=False)
        pts = pts[idx]

    return pts


# ─────────────────────────────────────────────────────────────
def _run_inference(model, tensor: torch.Tensor, is_full: bool):
    """
    Chạy inference trên 1 model, trả về logits và critical_indices.

    Args:
        model: PointNetBasic hoặc PointNetFull
        tensor: (1, N, 3)
        is_full: True nếu là PointNetFull (có T-Net)

    Returns:
        probs: (num_classes,) — probabilities
        critical_indices: (1024,) — critical point indices
    """
    with torch.no_grad():
        if is_full:
            logits, critical_indices, _ = model(tensor)
        else:
            logits, critical_indices = model(tensor)

        probs = F.softmax(logits, dim=1)[0]  # (num_classes,)
        critical_indices = critical_indices[0]  # (1024,)

    return probs.numpy(), critical_indices.numpy()


def classify(
    points: list | np.ndarray,
    num_points: int = DEFAULT_NUM_POINTS,
    rotation_x: float = 0.0,
    rotation_y: float = 0.0,
    rotation_z: float = 0.0,
    noise_level: float = 0.0,
    drop_ratio: float = 0.0,
) -> dict:
    """
    Phân loại point cloud 3D bằng cả PointNet Basic và Full.

    Args:
        points: list hoặc numpy array hình dạng (N, 3)
        num_points: số điểm để sample (64/128/256/512/1024)

    Returns:
        dict với structure:
            basic_model: {label, class_id, confidence, top3}
            full_model:  {label, class_id, confidence, top3}
            critical_points: [(x,y,z), ...] — tọa độ của critical points
            num_points_used: int
            processing_time_ms: float
    """
    load_models_if_needed()
    class_names = _load_class_names()

    start = time.time()

    # ── Preprocess ──
    pts_raw = np.array(points, dtype=np.float32)

    # ── Áp dụng perturbation TRƯỚC khi normalize/sample ──
    pts_raw = apply_perturbation(pts_raw, rotation_x, rotation_y, rotation_z, noise_level, drop_ratio)

    pts_np = preprocess_point_cloud(pts_raw, num_points)   # (1, num_points, 3)
    pts_tensor = torch.from_numpy(pts_np).float()          # tensor (1, num_points, 3)

    # ── Basic model inference ──
    basic_probs, basic_crit_idx = _run_inference(_basic_model, pts_tensor, is_full=False)

    # ── Full model inference ──
    full_probs, full_crit_idx = _run_inference(_full_model, pts_tensor, is_full=True)

    # ── Build top-3 results ──
    def build_result(probs, crit_idx, pts_sampled, model_label):
        top3_idx = np.argsort(probs)[::-1][:3]
        top3 = [
            {
                "label": class_names[i],
                "class_id": int(i),
                "confidence": float(probs[i])
            }
            for i in top3_idx
        ]
        pred_id = int(top3_idx[0])

        # ── Critical points: lấy unique indices từ argmax ──
        unique_crit = np.unique(crit_idx)
        # Trả về tọa độ của critical points trong point cloud đã sample
        crit_coords = pts_sampled[unique_crit].tolist()  # list of [x,y,z]

        return {
            "label": class_names[pred_id],
            "class_id": pred_id,
            "confidence": float(probs[pred_id]),
            "top3": top3,
            "critical_points": crit_coords,
            "num_critical": len(unique_crit)
        }

    pts_sampled = pts_np[0]  # (num_points, 3)

    basic_result = build_result(basic_probs, basic_crit_idx, pts_sampled, "basic")
    full_result  = build_result(full_probs,  full_crit_idx,  pts_sampled, "full")

    elapsed_ms = (time.time() - start) * 1000

    return {
        "basic_model": basic_result,
        "full_model": full_result,
        "point_cloud": pts_sampled.tolist(),   # toàn bộ điểm đã normalize (cho 3D render)
        "num_points_used": num_points,
        "processing_time_ms": round(elapsed_ms, 2)
    }


# ─────────────────────────────────────────────────────────────
# Sample clouds loader
# ─────────────────────────────────────────────────────────────
def load_sample_cloud(class_name: str) -> np.ndarray | None:
    """
    Load file .npy mẫu theo tên class.

    Args:
        class_name: ví dụ 'airplane', 'chair', 'car', 'lamp', 'table'

    Returns:
        numpy array (2048, 3) hoặc None nếu không tìm thấy
    """
    path = os.path.join(SAMPLE_CLOUDS, f"{class_name}_sample.npy")
    if not os.path.exists(path):
        return None
    return np.load(path)


def get_available_samples() -> list[str]:
    """
    Trả về danh sách class có file sample sẵn.

    Returns:
        list tên class, ví dụ ['airplane', 'chair', ...]
    """
    if not os.path.exists(SAMPLE_CLOUDS):
        return []
    files = [f for f in os.listdir(SAMPLE_CLOUDS) if f.endswith('_sample.npy')]
    return [f.replace('_sample.npy', '') for f in sorted(files)]


def get_all_classes() -> list[str]:
    """Trả về danh sách 40 class của ModelNet40."""
    return _load_class_names()
