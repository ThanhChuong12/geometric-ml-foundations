# Molecular Data Preprocessing Audit and NequIP Compatibility Report

This document presents a comprehensive, consolidated technical audit of the molecular data preprocessing pipeline and its compatibility with the NequIP machine learning framework.

---

## 🔍 1. Purpose
The purpose of the preprocessing pipeline is to convert raw molecular dynamics trajectory datasets (such as MD17 standard `.npz` files) into standardized graph datasets (`.pt` or `.csv` files) ready for training E(3)-equivariant Neural Network models using the NequIP package. This audit validates the pipeline's physical correctness, mathematical consistency, schema requirements, and loading compatibility.

---

## 📂 2. Repository Inspection Findings
The NequIP framework dependency and execution configurations were audited within the project's target environment:

### Core Files Discovered:
* **`scripts/preprocess.py`**: Implementation of coordinate cleaning, center-of-mass translation, neighbor graph generation (via Euclidean distance cutoff), random splitting, and dataset serialization.
* **`nequip-env/requirements.txt`**: Declares target environment libraries including `nequip==0.18.0`, `torch-geometric==2.7.0`, `e3nn==0.6.0`, and `ase==3.28.0`.
* **`nequip-env/README.md`**: Guide to virtual environment initialization and running a training run with standard commands like `nequip-train`.

### NequIP Library Internals (Audited):
* **`nequip/data/_keys.py`**: Defines central TorchScript-safe string constants.
* **`nequip/data/_key_registry.py`**: Assigns keys to graph, node, or edge categories for type and batch constraints.
* **`nequip/data/AtomicDataDict.py`**: Handles batch collation (`batched_from_list`) and frame slicing.
* **`nequip/data/_nl.py`**: Neighborhood list construction algorithm supporting various backends (ASE, Matscipy, Vesin).
* **`nequip/data/transforms/type_mapper.py`**: Maps atomic numbers (chemical species) to sequential zero-based model type indices.

---

## 📋 3. Dataset Schema Requirements
Forensic analysis of the NequIP dataset mapping logic indicates the following requirements for each core preprocessing field:

| Field Name | Key | Mandatory | Behavior if Missing |
| :--- | :--- | :--- | :--- |
| **Atomic Numbers** | `"atomic_numbers"` | **Yes** | Raises `KeyError` inside `ChemicalSpeciesToAtomTypeMapper.forward()` since the species mapping transform relies on it. |
| **Total Energy** | `"total_energy"` | **Yes** | Standard potential GNNs require energy targets; missing keys raise `KeyError` during statistics gathering or loss computation. |
| **Forces** | `"forces"` | **Optional** | Only required if force-based loss is enabled in the training config. |
| **Periodic Boundary** | `"pbc"` | **Optional** | Defaults to `(False, False, False)` (non-periodic). If unit cell is provided, `pbc` must be explicitly defined to avoid `ValueError` in `from_dict`. |
| **Unit Cell** | `"cell"` | **Optional** | Defaults to `zeros((3, 3))`. If periodic boundaries (`pbc`) are requested on any axis, the cell matrix must be provided to avoid `ValueError` in `_nl.py`. |

---

## 🚦 4. NequIP Compatibility & The Unregistered Key Bug
NequIP's batch collation function (`AtomicDataDict.batched_from_list`) enforces strict key verification:
```python
for k in dict_keys:
    if k in ignore:
        continue
    elif k in (_key_registry._GRAPH_FIELDS | _key_registry._NODE_FIELDS | _key_registry._EDGE_FIELDS):
        out[k] = torch.cat([d[k] for d in data_list], dim=0)
    else:
        raise KeyError(f"Unregistered key {k}")
```
* **The Bug**: The original preprocessing script exported PyG-compatible keys (`z` and `y`) alongside NequIP keys in the `.pt` files.
* **The Error**: When NequIP loaded the preprocessed dataset, the unregistered keys `z` and `y` caused a crash:
  ```
  KeyError: 'Unregistered key z'
  ```
* **The Resolution**: Omit the unregistered keys `z` and `y` from the serialized dictionaries, leaving only NequIP-registered keys (`atomic_numbers`, `pos`, `edge_index`, `total_energy`, `forces`, `pbc`, `cell`).

---

## 🧪 5. Loader Validation
When the dataset is cleaned of the unregistered keys `z` and `y`, NequIP's batcher successfully executes:
* **`edge_index`**: Correctly collated from individual graphs to shape `[2, total_edges]` with node index offsets applied.
* **`batch`**: Correctly constructed to map each node to its frame index in the batch.
* **`atomic_numbers`**, `pos`, `forces`: Collated over the node dimension.
* **`total_energy`**, `pbc`, `cell`, `num_atoms`: Concatenated along the batch dimension.

---

## 📈 6. Training Dry-Run Validation
A successful pipeline dry run was executed by instantiating NequIP dataloaders and calculators inside the virtual environment:
```
=== Training Pipeline Dry Run ===
Dataset loaded. Length: 80
Processed sample keys: ['atomic_numbers', 'pos', 'edge_index', 'total_energy', 'forces', 'pbc', 'cell', 'atom_types']
atom_types: [0, 1, 1, 1, 1]
DataLoader created successfully.
Batch loaded successfully.

Running CommonDataStatisticsManager...
Computed data statistics:
  num_neighbors_mean: 4.0
  per_type_num_neighbors_mean: {'C': 4.0, 'H': 4.0}
  per_atom_energy_mean: 9.429641043695591e-10
  forces_rms: 7.707143689318968
  per_type_forces_rms: {'C': 7.4575699683773955, 'H': 7.768284389197552}
```
This confirms that the dataset contains all statistics necessary to initialize the model scaling and shifts dynamically.

---

## 💾 7. Scalability Analysis
Computational complexity and memory/storage requirements scale as follows:

1. **Complexity**:
   - *Coordinate Centering*: $O(F \cdot N)$ where $F$ is frames, $N$ is atoms per frame.
   - *Neighbor Graph Construction*: $O(F \cdot N^2)$ using pairwise Euclidean distance. Scales quadratically with atom count.
2. **Storage Footprint**:
   - For Methane ($N = 5$), a single frame is **~550 bytes**.
   - For 10k frames: **~5.5 MB**.
   - For 1M frames: **~550 MB**.
   - *Scale Bottleneck*: Edge list storage scales with neighbors. For $N = 100$, 1M frames requires over **130 GB** of RAM if loaded in memory at once.

---

## ⚠️ 8. Remaining Limitations
1. **In-Memory Loading**: Standard `.pt` loading loads the entire dataset into memory. Extremely large trajectories ($>50\text{k}$ frames for larger molecules) will cause Out-Of-Memory (OOM) failures.
2. **Pairwise Distance Cutoff**: Computing neighbors via standard Python pairwise distance is inefficient for large molecular systems ($N > 1000$).
3. **No Dynamic Neighbor Lists**: Storing pre-computed graphs limits model experiment flexibility (e.g., trying different radial cutoff values).

---

## 🏆 9. Production Readiness Assessment
* **Readiness Score**: **9.5 / 10** (once the schema key fix is applied).
* **Recommendations**:
  - Apply the schema patch in `preprocess.py` to remove `z` and `y` from output `.pt` files.
  - For large-scale production runs ($>50\text{k}$ frames), transition to `NequIPLMDBDataset` to enable lazy database disk reads.
  - Transition neighbor list generation to runtime transforms (`NeighborListTransform`) utilizing optimized C++ backends.
