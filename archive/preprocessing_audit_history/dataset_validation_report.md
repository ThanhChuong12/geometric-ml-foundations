# Dataset Validation Report

This report evaluates the preprocessing pipeline against real-world molecular datasets (such as MD17, revised MD17/rMD17, and QM9).

---

## 🔍 1. Repository Search Summary

A recursive search of the workspace was conducted for the following dataset-related terms:
- `MD17`
- `revised MD17`
- `rMD17`
- `NequIP example datasets`
- `molecular trajectory datasets`

**Finding**: No real datasets (e.g., `.npz` files or `.extxyz` files) are stored in the repository. The only dataset present is the synthetic methane trajectory (`data/raw/mock_md17.npz`) generated for testing purposes. Therefore, the pipeline has not been executed on real molecular trajectories.

---

## 📋 2. Expected Dataset Structures

To ensure compatibility when a user downloads a real dataset, the preprocessing script expects specific structures:

### A. MD17 / rMD17 Dataset Structure (NumPy `.npz`)
Standard MD17 trajectories (e.g., aspirin, benzene, ethanol) contain:
- **`z`**: Array of shape `(N_atoms,)` (nuclear charges, `int`).
- **`R`**: Array of shape `(N_frames, N_atoms, 3)` (coordinates in Angstroms, `float`).
- **`E`**: Array of shape `(N_frames, 1)` or `(N_frames,)` (potential energies in kcal/mol or eV, `float`).
- **`F`**: Array of shape `(N_frames, N_atoms, 3)` (forces in kcal/(mol·Å) or eV/Å, `float`).

> [!NOTE]
> Revised MD17 (rMD17) provides forces and energies computed at higher precision, but retains the identical dictionary key structure (`z`, `R`, `E`, `F`).

### B. QM9 Dataset Structure (Alternative)
QM9 contains stationary structures of small organic molecules. Preprocessing QM9 for GNNs requires:
- Reading coordinate coordinates and species from `.xyz` or `.sdf` source files.
- Extracted features include: atomic numbers, positions, and quantum chemical targets (e.g., dipole moment $\mu$, isotropic polarizability $\alpha$, HOMO/LUMO, total energy $U_0$ at $0\text{ K}$).

---

## 📝 3. Dataset Validation Checklist

When loading a real dataset, the following checklist must be satisfied:

- [ ] **File Integrity**: The `.npz` archive can be successfully loaded via `np.load` without throwing `ZipError` or corrupted header exceptions.
- [ ] **Required Keys**: The `.npz` file contains the exact keys `'z'`, `'R'`, `'E'`, and `'F'`.
- [ ] **Array Types & Precision**:
  - `z` is a 1D integer array.
  - `R` and `F` are float32/float64 3D arrays.
  - `E` is a float32/float64 1D/2D array.
- [ ] **Shape Consistency**:
  - `len(z) == R.shape[1] == F.shape[1]` (atomic counts match).
  - `R.shape[0] == E.shape[0] == F.shape[0]` (number of frames matches).
  - `R.shape[2] == F.shape[2] == 3` (3D Cartesian coordinates).
- [ ] **Value Sanity**:
  - Coordinate coordinates do not contain `NaN` or `Inf`.
  - Atomic numbers `z` are valid (> 0 and correspond to chemical elements).
  - Bond distances between adjacent atoms are within physically plausible bounds ($0.5\text{ \AA}$ to $3.0\text{ \AA}$).

---

## ⚠️ 4. Unverified Aspects

Due to the absence of real molecular trajectory files in this workspace, the following behaviors remain unverified:
1. **Memory Overhead with Large Trajectories**: Real MD17 datasets can contain up to $100,000$ to $1,000,000$ frames. Loading the entire dataset into memory as numpy arrays might cause Out-Of-Memory (OOM) issues on systems with low RAM.
2. **Physical Cutoff Sensitivity**: For larger molecules (like aspirin, which has 21 atoms, or azobenzene, which has 24 atoms), a cutoff distance of $4.0\text{ \AA}$ will yield complex connectivity and a larger number of edges compared to methane. The graph size scaling and neighbor statistics have not been tested.
3. **Double Precision Scaling**: Real datasets may store arrays in `float64`. The script's conversion to `float32` tensors needs to be checked for potential precision loss or casting issues.
