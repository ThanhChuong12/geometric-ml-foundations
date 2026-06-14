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

    if len(request.points) < 10:
        raise HTTPException(
            status_code=400,
            detail="Point cloud too few points. Need at least 10 points."
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
    cloud = load_sample_cloud(class_name)

    if cloud is None:
        available = get_available_samples()
        raise HTTPException(
            status_code=404,
            detail=(
                f"Sample does not exist for class '{class_name}'. "
                f"Available: {available}. "
                f"Run: python scripts/download_assets.py"
            )
        )

    return SampleCloudResponse(
        class_name=class_name,
        points=cloud.tolist(),
        num_points=len(cloud)
    )


@router.get("/classes", response_model=ClassListResponse)
async def list_classes():
    classes = get_all_classes()
    return ClassListResponse(classes=classes, total=len(classes))
