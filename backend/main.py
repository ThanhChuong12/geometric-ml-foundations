# backend/main.py

import os
# Force CPU only (hide GPU to prevent CUDA driver mismatch/initialization errors)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.part1_routes import router as part1_router
from api.part2_routes import router as part2_router
from api.part3_routes import router as part3_router
from core.nequip_wrapper import NequIPPredictor
from services.qm9_service import MoleculeGraphService

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("main")

# Resolve absolute path to NequIP best checkpoint dynamically
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
CHECKPOINT_PATH = REPO_ROOT / "ai_core" / "outputs" / "nequip_l1_1000" / "best.ckpt"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager handling startup and shutdown tasks."""
    # --- Startup Lifecycle ---
    checkpoint_path = str(CHECKPOINT_PATH)
    logger.info(f"Starting server. Checking NequIP checkpoint at '{checkpoint_path}'...")
    
    predictor = NequIPPredictor(checkpoint_path=checkpoint_path, device="cpu")
    
    try:
        # Pre-load the model eagerly during startup to catch schema/checkpoint errors early
        predictor.load_model()
        app.state.predictor = predictor
        
        # Instantiate and cache the MoleculeGraphService using loaded metadata
        app.state.graph_service = MoleculeGraphService(
            type_names=predictor.type_names,
            r_max=predictor.r_max
        )
        logger.info("NequIP backend services initialized successfully on CPU.")
    except Exception as e:
        logger.critical(
            f"CRITICAL: Failed to initialize NequIP predictor on startup: {str(e)}", 
            exc_info=True
        )
        # Do not crash the server lifecycle, but keep state None to indicate degraded mode
        app.state.predictor = None
        app.state.graph_service = None

    yield
    # --- Shutdown Lifecycle ---
    logger.info("Shutting down molecular energy prediction API server.")


app = FastAPI(
    title="Molecular Energy Prediction API",
    description="Production-quality FastAPI backend for NequIP-based E(3)-equivariant inference on QM9.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production to specify trusted origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Router registration
app.include_router(part1_router, prefix="/api/part1")
app.include_router(part2_router, prefix="/api/part2")
app.include_router(part3_router, prefix="/api/part3")


@app.get("/health", summary="Health check endpoint")
def health_check():
    """Returns the health status of the application and its model dependencies."""
    predictor = getattr(app.state, "predictor", None)
    if predictor is None or predictor._model is None:
        return {"status": "degraded", "detail": "NequIP predictor not loaded"}
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
