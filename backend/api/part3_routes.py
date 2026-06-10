# backend/api/part3_routes.py

import logging
from fastapi import APIRouter, Depends, HTTPException, Request, status
from schemas.prediction import MoleculeInput, EnergyResponse
from core.nequip_wrapper import NequIPPredictor, NequIPException
from services.qm9_service import MoleculeGraphService

logger = logging.getLogger("part3_routes")
router = APIRouter()


def get_predictor(request: Request) -> NequIPPredictor:
    """Dependency helper to retrieve the predictor instance from app state."""
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="NequIP Predictor Service is not initialized."
        )
    return predictor


def get_graph_service(request: Request) -> MoleculeGraphService:
    """Dependency helper to retrieve the graph conversion service from app state."""
    graph_service = getattr(request.app.state, "graph_service", None)
    if graph_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Molecule Graph Service is not initialized."
        )
    return graph_service


@router.post(
    "/predict-energy", 
    response_model=EnergyResponse, 
    status_code=status.HTTP_200_OK,
    summary="Predict molecular energy",
    description="Accepts molecular structure (elements and coordinates) and runs NequIP energy-only inference."
)
def predict_energy(
    payload: MoleculeInput,
    predictor: NequIPPredictor = Depends(get_predictor),
    graph_service: MoleculeGraphService = Depends(get_graph_service)
):
    """Energy-only prediction route."""
    logger.info(
        f"Received prediction request with {len(payload.atomic_numbers)} atoms."
    )
    try:
        # 1. Transform structure to graph dictionary
        graph = graph_service.create_graph(
            atomic_numbers=payload.atomic_numbers,
            positions=payload.positions
        )

        # 2. Run model prediction
        energy = predictor.predict(graph)

        logger.info(f"Successful prediction. Energy: {energy:.6f}")
        return EnergyResponse(energy=energy)

    except NequIPException as e:
        logger.error(f"Inference exception: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference processing failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected exception during request handling: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected server error: {str(e)}"
        )
