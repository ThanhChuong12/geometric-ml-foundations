# Schema Verification Report

This report presents a verified schema analysis for the core preprocessing fields based on actual NequIP library code.

---

## 🔍 Verification Matrix

### 1. `atomic_numbers` (Key: `"atomic_numbers"`)
* **Where is it referenced?**
  - In `nequip/data/transforms/type_mapper.py` inside `ChemicalSpeciesToAtomTypeMapper.forward()`:
    ```python
    atomic_numbers = data[AtomicDataDict.ATOMIC_NUMBERS_KEY]
    data[AtomicDataDict.ATOM_TYPE_KEY] = torch.index_select(
        self.lookup_table, 0, atomic_numbers
    )
    ```
* **Is it mandatory?** Yes. NequIP models work with abstract `atom_types` indices. The `TypeMapper` transform (run during training initialization) translates chemical symbols to internal indices using `atomic_numbers`.
* **Is it optional?** No.
* **What happens if it is missing?** Throws a `KeyError: 'atomic_numbers'` inside `type_mapper.py` when indexing the dictionary.

---

### 2. `total_energy` (Key: `"total_energy"`)
* **Where is it referenced?**
  - In `nequip/data/dataset/npz_dataset.py` during initialization:
    ```python
    for k, v in key_mapping.items():
        if v == AtomicDataDict.TOTAL_ENERGY_KEY:
            E_key = k
    assert E_key is not None, "No key corresponding to `total_energy` found in npz dataset"
    ```
  - In `nequip/data/stats_manager.py` during dataset statistics calculation.
* **Is it mandatory?** Yes. Standard potential energy GNNs require energy targets for training and validation.
* **Is it optional?** Only optional if executing a specialized force-only or property-only model. For standard workflows, it is mandatory.
* **What happens if it is missing?** Raises a `KeyError: 'total_energy'` during loss calculation or statistics gathering.

---

### 3. `forces` (Key: `"forces"`)
* **Where is it referenced?**
  - In `nequip/nn/grad_output.py` and loss calculators when calculating force prediction errors.
* **Is it mandatory?** Optional.
* **Is it optional?** Yes, if the network is trained solely on energy (energy-only training).
* **What happens if it is missing?** If force training is enabled in the training config, the loader/trainer will attempt to retrieve `"forces"` to compute loss, raising a `KeyError`. If force training is disabled, the script runs safely without it.

---

### 4. `pbc` (Key: `"pbc"`)
* **Where is it referenced?**
  - In `nequip/data/dict.py` (lines 16-20):
    ```python
    cell = data.get(AtomicDataDict.CELL_KEY, None)
    pbc = data.get(AtomicDataDict.PBC_KEY, None)
    if pbc is None:
        if cell is not None:
            raise ValueError(
                "A cell was provided, but pbc's were not. Please explicitly provide PBC."
            )
        pbc = False
    ```
  - In `nequip/data/_nl.py` in `_compute_neighborlist_single_frame` (lines 124-131).
* **Is it mandatory?** Optional.
* **Is it optional?** Yes.
* **What happens if it is missing?**
  - If both `pbc` and `cell` are missing, `pbc` defaults to `False` (non-periodic, `(False, False, False)`), and neighbor list generation proceeds without periodic boundaries.
  - **CRITICAL EXCEPTION**: If `cell` is provided but `pbc` is missing, `nequip/data/dict.py` raises `ValueError: A cell was provided, but pbc's were not. Please explicitly provide PBC.`.

---

### 5. `cell` (Key: `"cell"`)
* **Where is it referenced?**
  - In `nequip/data/dict.py` (lines 14-25) and `nequip/data/_nl.py` (lines 123-132):
    ```python
    if cell is not None:
        temp_cell = cell.detach().cpu().numpy()
    else:
        if pbc[0] or pbc[1] or pbc[2]:
            raise ValueError(
                "Periodic boundary conditions requested but no cell was provided."
            )
        temp_cell = np.zeros((3, 3), dtype=temp_pos.dtype)
    ```
* **Is it mandatory?** Optional.
* **Is it optional?** Yes.
* **What happens if it is missing?**
  - If both are missing, `cell` defaults to `zeros((3, 3))`.
  - **CRITICAL EXCEPTION**: If any `pbc` axis is set to `True`, but `cell` is missing, `nequip/data/_nl.py` raises `ValueError: Periodic boundary conditions requested but no cell was provided.`.
