"""
Part 2 API Routes — PointNet 3D Classification Demo.

Endpoints:
    POST /api/part2/classify     — Classify a point cloud
    GET  /api/part2/sample/{cls} — Get sample point cloud by class name
    GET  /api/part2/classes      — List all 40 ModelNet40 classes
    GET  /api/part2/status       — Health check

Không ảnh hưởng đến Part 1 routes.
"""

from fastapi import APIRouter, HTTPException
from models.pointnet_schemas import (
    PointCloudRequest,
    PointNetResponse,
    SampleCloudResponse,
    ClassListResponse,
    StatusResponse,
)
from services.pointnet_service import (
    classify,
    load_sample_cloud,
    get_available_samples,
    get_all_classes,
    load_models_if_needed,
)

router = APIRouter()


@router.get("/status", response_model=StatusResponse)
async def pointnet_status():
    """
    Health check endpoint.
    Trả về trạng thái model và danh sách sample có sẵn.
    """
    loaded = load_models_if_needed()
    samples = get_available_samples()

    return StatusResponse(
        status="ok",
        models_loaded=loaded,
        available_samples=samples,
        message=(
            "PointNet models loaded successfully."
            if loaded
            else "Models not loaded. Run: python scripts/download_assets.py"
        )
    )


@router.post("/classify", response_model=PointNetResponse)
async def classify_point_cloud(request: PointCloudRequest):
    """
    Phân loại point cloud 3D bằng PointNet Basic và Full.

    Body:
        points: list of [x, y, z] — tối thiểu 64 điểm
        num_points: số điểm để sample (64/128/256/512/1024)

    Response:
        basic_model: kết quả từ PointNet không có T-Net
        full_model:  kết quả từ PointNet đầy đủ (có T-Net)
        point_cloud: điểm sau normalize (dùng cho 3D render)
        critical_points: điểm critical theo Theorem 2
    """
    if len(request.points) < 10:
        raise HTTPException(
            status_code=400,
            detail="Point cloud quá ít điểm. Cần ít nhất 10 điểm."
        )

    try:
        result = classify(
            request.points,
            request.num_points,
            rotation_x=request.rotation_x,
            rotation_y=request.rotation_y,
            rotation_z=request.rotation_z,
            noise_level=request.noise_level,
            drop_ratio=request.drop_ratio,
        )
        return PointNetResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")


@router.get("/sample/{class_name}", response_model=SampleCloudResponse)
async def get_sample_point_cloud(class_name: str):
    """
    Lấy point cloud mẫu theo tên class.

    Path params:
        class_name: tên class, ví dụ 'airplane', 'chair', 'car', 'lamp', 'table'

    Response:
        class_name: tên class
        points: list of [x, y, z] — 2048 điểm gốc từ ModelNet40
        num_points: số lượng điểm
    """
    cloud = load_sample_cloud(class_name)

    if cloud is None:
        available = get_available_samples()
        raise HTTPException(
            status_code=404,
            detail=(
                f"Sample không tồn tại cho class '{class_name}'. "
                f"Có sẵn: {available}. "
                f"Chạy: python scripts/download_assets.py"
            )
        )

    return SampleCloudResponse(
        class_name=class_name,
        points=cloud.tolist(),
        num_points=len(cloud)
    )


@router.get("/classes", response_model=ClassListResponse)
async def list_classes():
    """
    Trả về danh sách đầy đủ 40 class của ModelNet40.
    """
    classes = get_all_classes()
    return ClassListResponse(classes=classes, total=len(classes))
