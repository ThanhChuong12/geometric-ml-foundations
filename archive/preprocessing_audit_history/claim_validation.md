# Claim Validation Report

This report evaluates key assertions from previous audits against direct code forensics and execution results.

---

## 📊 Claim Assessments

### 1. Claim: "NequIP requires a custom class `AtomicData` to wrap datasets."
* **Audit Status**: **FALSE**
* **Evidence**:
  - Forensics of `nequip/data/AtomicDataDict.py` shows that there is no custom `AtomicData` class defined in this NequIP version.
  - The dataset represents graphs as standard Python dictionaries, aliased using `AtomicDataDict.Type = Dict[str, torch.Tensor]`.
  - The loader collation function (`AtomicDataDict.batched_from_list`) operates directly on lists of Python dictionaries.

---

### 2. Claim: "NequIP requires standard PyTorch Geometric keys `z` and `y` to load data."
* **Audit Status**: **FALSE**
* **Evidence**:
  - In `nequip/data/AtomicDataDict.py` (lines 135-139), any key not registered in NequIP's `_key_registry.py` raises `KeyError(f"Unregistered key {k}")`.
  - Since standard PyG keys `z` and `y` are not registered, passing them directly to NequIP's collation loader raises a `KeyError: 'Unregistered key z'`.
  - NequIP expects `"atomic_numbers"` and `"total_energy"` instead of `z` and `y`.

---

### 3. Claim: "The neighborhood edge index must be pre-computed during preprocessing to train with NequIP."
* **Audit Status**: **PARTIALLY VERIFIED**
* **Evidence**:
  - Pre-computing `edge_index` is fully supported by NequIP if the dictionary keys are clean.
  - However, NequIP's standard configuration (`tutorial.yaml`) demonstrates that neighbor lists are typically calculated dynamically at runtime using `NeighborListTransform` inside the dataset loader pipeline. Thus, pre-computing `edge_index` is not strictly mandatory.

---

### 4. Claim: "Periodic boundary conditions (`pbc`) and unit cells (`cell`) are mandatory fields."
* **Audit Status**: **FALSE**
* **Evidence**:
  - If both `pbc` and `cell` are missing, NequIP applies safe default values: `pbc = (False, False, False)` and `cell = zeros((3, 3))`.
  - **Dependency constraints**:
    1. If `cell` is provided, `pbc` must be explicitly provided (checked in `nequip/data/dict.py`).
    2. If `pbc` is `True` on any axis, `cell` must be provided (checked in `nequip/data/_nl.py`).
    Otherwise, `ValueError` is thrown.

---

### 5. Claim: "Atomic coordinates must be centered at the center-of-mass prior to model training."
* **Audit Status**: **FALSE**
* **Evidence**:
  - NequIP's `ASEDataset`, `NPZDataset`, and standard model configuration files do not require coordinates to be centered.
  - While centering is a highly recommended practice to reduce translational variance, it is not enforced by any library loaders or transforms.
