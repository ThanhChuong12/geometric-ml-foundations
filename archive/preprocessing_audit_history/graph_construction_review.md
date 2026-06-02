# Graph Construction Review

This report provides an in-depth review of the graph connectivity construction function (`compute_edge_index()`) used in the preprocessing pipeline.

---

## 🔍 1. Edge Index Calculation & Logic

The connectivity of the molecular graphs is determined using a radial cutoff distance ($R_c$) strategy. Given the atomic coordinate matrix $\mathbf{r} \in \mathbb{R}^{N \times 3}$:
1. **Pairwise Differences**: The differences between atom coordinates are computed:
   $$\mathbf{D}_{ij} = \mathbf{r}_i - \mathbf{r}_j \in \mathbb{R}^{N \times N \times 3}$$
2. **Euclidean Distances**: Pairwise Euclidean distances are computed:
   $$d_{ij} = \|\mathbf{D}_{ij}\|_2 = \sqrt{\sum_{k} (r_{i,k} - r_{j,k})^2} \in \mathbb{R}^{N \times N}$$
3. **Adjacency Masking**: Directed edges are created for pairs matching:
   $$e_{ij} \in \mathcal{E} \iff d_{ij} \le R_c \quad \text{and} \quad i \neq j$$

---

## 📊 2. Cutoff Sensitivity Analysis (Methane Example)

Methane ($\text{CH}_4$, 5 atoms) has a central Carbon atom (C) and four surrounding Hydrogen atoms (H). The typical equilibrium bond lengths are:
- C-H bond length: $\approx 1.09\text{ \AA}$
- H-H distance: $\approx 1.78\text{ \AA}$

Depending on the cutoff radius $R_c$, the graph structure transitions as follows:

| Cutoff Radius $R_c$ ($\text{\AA}$) | Connectivity State | Directed Edge Count | Avg. Neighbors per Atom | Description |
|---|---|---|---|---|
| $R_c < 1.0$ | Disconnected | 0 | 0.0 | All atoms are isolated. |
| $1.0 \le R_c < 1.7$ | Partially Connected | 8 | 1.6 | Only C-H and H-C bonds are active. |
| $R_c \ge 1.8$ | Fully Connected | 20 | 4.0 | C-H and H-H interactions are all active. |

---

## ⚠️ 3. Edge Case & Failure Handling

1. **Self-loops**:
   - *Handling*: Handled by masking out the diagonal using `~np.eye(num_atoms, dtype=bool)`. This prevents self-loops (`i == i`).
2. **Disconnected Graphs**:
   - *Handling*: If $R_c$ is set too small, no edges are generated. The pipeline will output a graph with an empty `edge_index` tensor of shape `[2, 0]`.
   - *NequIP Sensitivity*: NequIP models require message passing; an empty edge list will cause internal division-by-zero or shape errors during convolution steps.
3. **Empty Neighbor Lists / Isolated Atoms**:
   - *Handling*: If some atoms are isolated (e.g. in a gas phase simulation with a remote molecule, or if $R_c$ is small), their neighbor count is 0.
   - *NequIP Sensitivity*: Isolated atoms do not receive updates from neighboring atoms, which may cause gradient or statistical anomalies.
4. **Non-Finite Coordinate Edge Cases**:
   - *Risk*: If atomic coordinates contain extreme values, the pairwise distance calculation could overflow or produce `NaN`.
   - *Handling*: The script verifies that the computed pairwise distance matrix contains only finite float values. It throws an error if any `NaN` or `Inf` is generated.
