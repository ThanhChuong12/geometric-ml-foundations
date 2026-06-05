"""
Pydantic schemas cho PointNet Part 2 API.
Tách biệt hoàn toàn với schemas.py của Part 1.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class PointCloudRequest(BaseModel):
    """Request body để phân loại 1 point cloud."""
    points: List[List[float]] = Field(
        ...,
        description="Danh sách tọa độ [x, y, z]. Shape: (N, 3)"
    )
    num_points: int = Field(
        default=1024, ge=64, le=2048,
        description="Số điểm để sample. Tùy chọn: 64, 128, 256, 512, 1024"
    )
    # ── Perturbation params (thí nghiệm robustness) ──
    rotation_x: float = Field(default=0.0, ge=0.0, le=360.0, description="Góc xoay quanh trục X (độ)")
    rotation_y: float = Field(default=0.0, ge=0.0, le=360.0, description="Góc xoay quanh trục Y (độ)")
    rotation_z: float = Field(default=0.0, ge=0.0, le=360.0, description="Góc xoay quanh trục Z (độ)")
    noise_level: float = Field(default=0.0, ge=0.0, le=0.15, description="Mức nhiễu Gaussian (0 = không nhiễu, 0.1 = nhiễu mạnh)")
    drop_ratio: float = Field(default=0.0, ge=0.0, le=0.75, description="Tỷ lệ bỏ điểm ngẫu nhiên (0 = giữ nguyên, 0.5 = bỏ 50%)") 


class Top3Prediction(BaseModel):
    """Một entry trong top-3 predictions."""
    label: str
    class_id: int
    confidence: float


class ModelResult(BaseModel):
    """Kết quả classification của 1 model."""
    label: str            = Field(..., description="Top-1 predicted class name")
    class_id: int         = Field(..., description="Class index (0-39)")
    confidence: float     = Field(..., description="Top-1 confidence (0-1)")
    top3: List[Top3Prediction]
    critical_points: List[List[float]] = Field(
        ...,
        description="Coordinates of critical points that determine the global feature (Theorem 2)"
    )
    num_critical: int     = Field(..., description="Number of unique critical points")


class PointNetResponse(BaseModel):
    """Response trả về từ /classify endpoint."""
    basic_model: ModelResult   = Field(..., description="PointNet without T-Net (baseline)")
    full_model: ModelResult    = Field(..., description="PointNet with Input+Feature Transform")
    point_cloud: List[List[float]] = Field(
        ...,
        description="Preprocessed & normalized point cloud used for inference"
    )
    num_points_used: int
    processing_time_ms: float


class SampleCloudResponse(BaseModel):
    """Response trả về sample point cloud theo class."""
    class_name: str
    points: List[List[float]]
    num_points: int


class ClassListResponse(BaseModel):
    """Danh sách 40 class của ModelNet40."""
    classes: List[str]
    total: int


class StatusResponse(BaseModel):
    """Health check + trạng thái model."""
    status: str
    models_loaded: bool
    available_samples: List[str]
    message: str
