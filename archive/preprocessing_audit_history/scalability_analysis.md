# Scalability and Performance Analysis

This report analyzes the performance, scaling costs, and storage profiles of the molecular preprocessing pipeline for datasets containing 10k, 100k, and 1M frames.

---

## ⚡ 1. Computational Complexity & Scaling Laws

| Step | Algorithm | Complexity (Per Frame) | Complexity (Total Dataset) | Bottlenecks |
| :--- | :--- | :--- | :--- | :--- |
| **Coordinate Centering** | COM calculation | $O(N)$ | $O(F \cdot N)$ | Negligible. Linear in atom count. |
| **Neighbor Construction** | Pairwise distance cutoff | $O(N^2)$ (pairwise) | $O(F \cdot N^2)$ | High for large systems ($N > 1000$). Can be optimized to $O(N \log N)$ using KD-Trees/Cell-Lists. |
| **Serialization (.pt)** | PyTorch file writing | $O(N + E)$ | $O(F \cdot (N + E))$ | I/O-bound during storage operations. |

*Notation: $F$ is the number of frames, $N$ is the number of atoms per frame, and $E$ is the number of edges.*

---

## 💾 2. Storage & Memory Estimations

Based on the forensic audit of the preprocessed Methane dataset ($N = 5$, $E = 20$), a single frame takes approximately **550 bytes** in PyTorch tensor format.

### Estimated File Sizes:

| Split Size ($F$) | Binary `.pt` size | CSV (Wide/Long) size | Memory footprint (Full load) |
| :--- | :--- | :--- | :--- |
| **80 frames** (Current) | ~43 KB | ~28 KB | < 1 MB |
| **10k frames** | ~5.5 MB | ~3.5 MB | ~15 MB |
| **100k frames** | ~55 MB | ~35 MB | ~150 MB |
| **1M frames** | ~550 MB | ~350 MB | ~1.5 GB |

> [!NOTE]
> While a 1.5 GB memory footprint for 1M frames of Methane is easily handled on modern systems, memory scaling is **highly sensitive** to the number of atoms $N$. For a system with $N = 100$ atoms and $E = 5000$ edges, the storage footprint increases by ~250x, making 1M frames require over **130 GB** of RAM.

---

## 🚦 3. Bottlenecks & Architectural Recommendations

1. **Memory Exhaustion (OOM)**:
   - *Problem*: Loading massive `.pt` files (which store the entire list of dicts in memory) will eventually crash the system due to RAM limits.
   - *Solution*: For datasets $> 50\text{k}$ frames, utilize NequIP's **LMDB dataset** format (`NequIPLMDBDataset`). LMDB database backend enables lazy loading of individual frames from disk, keeping the memory footprint at $O(\text{batch\_size})$ rather than $O(F)$.

2. **Graph Construction Speed**:
   - *Problem*: Computing neighbor lists quadratically $O(N^2)$ per frame during preprocessing is slow.
   - *Solution*: Delegate neighbor list construction to NequIP's runtime transform (`NeighborListTransform`), which leverages optimized C++ backends (like `matscipy` or `vesin`). This avoids writing large edge lists to disk, saving up to 80% of storage space.

3. **CSV Export Overhead**:
   - *Problem*: Standard text-based CSV tables consume significant storage and are slow to parse compared to binary tensors.
   - *Solution*: Avoid CSV exports for production training runs. Restrict CSV formats solely to small debugging or inspection subsets.
