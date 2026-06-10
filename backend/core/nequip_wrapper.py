# backend/core/nequip_wrapper.py

import logging
import threading
from typing import Optional, List
import torch
from nequip.train.ema import EMALightningModule
from nequip.data import AtomicDataDict

logger = logging.getLogger("nequip_predictor")


class NequIPException(Exception):
    """Base exception for NequIP molecular energy predictor."""
    pass


class ModelLoadError(NequIPException):
    """Exception raised when loading the NequIP checkpoint fails."""
    pass


class InferenceError(NequIPException):
    """Exception raised when running inference fails."""
    pass


class NequIPPredictor:
    """Thread-safe, singleton-friendly wrapper around NequIP model for energy-only inference.

    Supports lazy loading and automatically handles device placement.
    """

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self._model: Optional[torch.nn.Module] = None
        self._type_names: Optional[List[str]] = None
        self._r_max: Optional[float] = None
        self._lock = threading.Lock()

    def load_model(self) -> None:
        """Load the model and hparams from checkpoint in a thread-safe manner."""
        if self._model is not None:
            return

        with self._lock:
            if self._model is not None:
                return

            logger.info(f"Loading NequIP model checkpoint from '{self.checkpoint_path}'...")
            try:
                # Load PyTorch Lightning module from checkpoint
                pl_module = EMALightningModule.load_from_checkpoint(
                    self.checkpoint_path, 
                    map_location=self.device
                )

                # Fetch model with EMA weights loaded
                sole_model = pl_module.evaluation_model["sole_model"]
                sole_model.to(self.device)
                sole_model.eval()

                # Extract metadata parameters for preprocessing
                # Fall back to checking hparams if attributes aren't directly available
                if hasattr(sole_model, "type_names"):
                    type_names = sole_model.type_names
                else:
                    type_names = pl_module.hparams["model"]["type_names"]

                if hasattr(sole_model, "r_max"):
                    r_max = sole_model.r_max
                else:
                    r_max = pl_module.hparams["model"]["r_max"]

                self._model = sole_model
                self._type_names = [str(t) for t in type_names]
                self._r_max = float(r_max)

                logger.info(
                    f"Successfully loaded NequIP model: r_max={self._r_max}, "
                    f"type_names={self._type_names}, device={self.device}"
                )
            except Exception as e:
                logger.error(f"Failed to load NequIP checkpoint: {str(e)}", exc_info=True)
                raise ModelLoadError(f"Could not load NequIP model from checkpoint: {str(e)}") from e

    @property
    def model(self) -> torch.nn.Module:
        """Get the loaded PyTorch model."""
        self.load_model()
        assert self._model is not None
        return self._model

    @property
    def type_names(self) -> List[str]:
        """Get the list of chemical type names registered in the model."""
        self.load_model()
        assert self._type_names is not None
        return self._type_names

    @property
    def r_max(self) -> float:
        """Get the cutoff radius of the model."""
        self.load_model()
        assert self._r_max is not None
        return self._r_max

    def predict(self, graph: AtomicDataDict.Type) -> float:
        """Run energy-only inference on the processed graph representation.

        Args:
            graph: The dictionary representation of the graph, containing POSITIONS_KEY,
                   ATOM_TYPE_KEY, EDGE_INDEX_KEY, and other required fields.

        Returns:
            The total molecular energy as a float.
        """
        self.load_model()
        assert self._model is not None

        try:
            # Move all input tensors to model device
            device_graph = {}
            for k, v in graph.items():
                if isinstance(v, torch.Tensor):
                    device_graph[k] = v.to(self.device)
                else:
                    device_graph[k] = v

            # Run energy-only inference under no_grad
            with torch.no_grad():
                output = self._model(device_graph)

                # Fetch total energy scalar
                if AtomicDataDict.TOTAL_ENERGY_KEY not in output:
                    raise KeyError(
                        f"Model output did not contain total energy key "
                        f"'{AtomicDataDict.TOTAL_ENERGY_KEY}'."
                    )

                energy_tensor = output[AtomicDataDict.TOTAL_ENERGY_KEY]
                energy = float(energy_tensor.cpu().item())

            return energy
        except Exception as e:
            logger.error(f"Error during NequIP model inference: {str(e)}", exc_info=True)
            raise InferenceError(f"NequIP inference failed: {str(e)}") from e