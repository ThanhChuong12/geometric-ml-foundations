from pydantic import BaseModel, Field
from typing import List, Optional

class PointCloudRequest(BaseModel):
    points: List[List[float]] = Field(
        ...,
        description="Coordinates of a point cloud. Shape: (N, 3)"
    )
    num_points: int = Field(
        default=1024, ge=64, le=2048,
        description="Number of points to sample. Options: 64, 128, 256, 512, 1024"
    )
    # Perturbation params
    rotation_x: float = Field(default=0.0, ge=0.0, le=360.0, description="Rotation around X axis (degrees)")
    rotation_y: float = Field(default=0.0, ge=0.0, le=360.0, description="Rotation around Y axis (degrees)")
    rotation_z: float = Field(default=0.0, ge=0.0, le=360.0, description="Rotation around Z axis (degrees)")
    noise_level: float = Field(default=0.0, ge=0.0, le=0.15, description="Gaussian noise level (0 = no noise, 0.1 = strong noise)")
    drop_ratio: float = Field(default=0.0, ge=0.0, le=0.75, description="Random point drop ratio (0 = no drop, 0.5 = 50% drop)") 


class Top3Prediction(BaseModel):
    label: str
    class_id: int
    confidence: float


class ModelResult(BaseModel):
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
    basic_model: ModelResult   = Field(..., description="PointNet without T-Net (baseline)")
    full_model: ModelResult    = Field(..., description="PointNet with Input+Feature Transform")
    point_cloud: List[List[float]] = Field(
        ...,
        description="Preprocessed & normalized point cloud used for inference"
    )
    num_points_used: int
    processing_time_ms: float


class SampleCloudResponse(BaseModel):
    class_name: str
    points: List[List[float]]
    num_points: int


class ClassListResponse(BaseModel):
    classes: List[str]
    total: int


class StatusResponse(BaseModel):
    status: str
    models_loaded: bool
    available_samples: List[str]
    message: str
