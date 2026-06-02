# Repository Forensics Report

This report documents the forensic search across the repository and its installed environment for files relating to NequIP datasets, configurations, loaders, and preprocessing fields.

---

## 📂 1. Repository Files

### A. Preprocessing Script
* **File Path**: `scripts/preprocess.py`
* **Purpose**: Primary data pipeline entry point. Performs coordinate cleaning, centering, neighbor graph construction, random split shuffling, target scaling, and file exports.
* **Relevant Lines**:
  - `validate_sample` (lines 51-115): Validates coordinates, atomic numbers, forces, and energies.
  - `compute_edge_index` (lines 131-156): Pairwise distance calculation and cutoff thresholding.
  - `build_graph_data` (lines 157-192): Packages the tensors into standard dictionaries.
* **Relationship to Preprocessing**: The implementation under audit.

### B. Environment & Setup Documentation
* **File Path**: `nequip-env/requirements.txt`
* **Purpose**: Declares exact dependency versions required for the NequIP environment (e.g., `nequip==0.18.0`, `torch-geometric==2.7.0`, `e3nn==0.6.0`, `ase==3.28.0`).
* **Relationship to Preprocessing**: Confirms the target package versions to verify compatibility.

* **File Path**: `nequip-env/README.md`
* **Purpose**: Guide to virtual environment configuration and running a demo training script.
* **Relevant Lines**:
  - `python scripts/train.py configs/example.yaml` (line 134)
* **Relationship to Preprocessing**: Identifies how NequIP model workflows are executed.

---

## 📦 2. Installed Environment Code (NequIP Library)

Since no training scripts are defined in the workspace, we audited the NequIP library code installed in the virtual environment at `nequip-env/.venv/Lib/site-packages/nequip/`.

### A. Key Registration Definitions
* **File Path**: `nequip-env/.venv/Lib/site-packages/nequip/data/_keys.py`
* **Purpose**: Centralizes the TorchScript-safe string constants representing dataset keys.
* **Relevant Lines**:
  - `POSITIONS_KEY: Final[str] = "pos"`
  - `ATOMIC_NUMBERS_KEY: Final[str] = "atomic_numbers"`
  - `TOTAL_ENERGY_KEY: Final[str] = "total_energy"`
  - `FORCE_KEY: Final[str] = "forces"`
  - `PBC_KEY: Final[str] = "pbc"`
  - `CELL_KEY: Final[str] = "cell"`
* **Relationship to Preprocessing**: Defines the exact naming convention required by NequIP.

* **File Path**: `nequip-env/.venv/Lib/site-packages/nequip/data/_key_registry.py`
* **Purpose**: Assigns keys to graph, node, or edge lists for dtype and batching constraints.
* **Relevant Lines**:
  - `_DEFAULT_LONG_FIELDS`: Includes `ATOMIC_NUMBERS_KEY` and `EDGE_INDEX_KEY`.
  - `_DEFAULT_GRAPH_FIELDS`: Includes `TOTAL_ENERGY_KEY`, `PBC_KEY`, and `CELL_KEY`.
  - `_DEFAULT_NODE_FIELDS`: Includes `POSITIONS_KEY`, `FORCE_KEY`, and `ATOMIC_NUMBERS_KEY`.
* **Relationship to Preprocessing**: Determines what keys are recognized during data collation.

### B. Batcher & Data Structure
* **File Path**: `nequip-env/.venv/Lib/site-packages/nequip/data/AtomicDataDict.py`
* **Purpose**: Manages the dictionary representation of graphs, handles single-frame batch initialization, and batches list elements.
* **Relevant Lines**:
  - `batched_from_list` (lines 78-142): Iterates over keys and raises `KeyError(f"Unregistered key {k}")` for any key not listed in `_key_registry`.
* **Relationship to Preprocessing**: Crucial constraint; explains why having PyG compatibility keys `z` and `y` in the preprocessed `.pt` file crashes NequIP's batch loading.

### C. Neighbor List Constructor
* **File Path**: `nequip-env/.venv/Lib/site-packages/nequip/data/_nl.py`
* **Purpose**: Constructs directed graph edge lists using `ase`, `matscipy`, or `vesin` backends.
* **Relevant Lines**:
  - `_compute_neighborlist_single_frame` (lines 92-192): Checks `pbc` and `cell`.
* **Relationship to Preprocessing**: Details how cell matrices and periodic boundary conditions are used to build neighbor lists.

### D. Species Index Mapper
* **File Path**: `nequip-env/.venv/Lib/site-packages/nequip/data/transforms/type_mapper.py`
* **Purpose**: Maps atomic numbers (chemical species) to sequential index positions required by the model.
* **Relevant Lines**:
  - `ChemicalSpeciesToAtomTypeMapper.forward` (lines 62-79): Reads `ATOMIC_NUMBERS_KEY` and outputs `ATOM_TYPE_KEY`. Raises `KeyError` if an atomic number is unmapped.
* **Relationship to Preprocessing**: Proves that `atomic_numbers` is mandatory for type mappings.
