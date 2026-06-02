# Production Readiness Review (v2)

This report evaluates the production readiness of the molecular preprocessing pipeline and provides recommendations for achieving maximum compatibility and performance.

---

## 📊 Score Summary

* **Current Score**: **7.0 / 10** (Functional, but output `.pt` files cause NequIP crash due to unregistered schema keys).
* **Target Score**: **10 / 10** (Once recommendations and schema updates are applied).

---

## 🔍 Issue Breakdown

### 🚨 1. Unregistered Schema Keys (`z`, `y`)
* **Impact**: **CRITICAL BLOCKER**
* **Details**: The preprocessing script outputs `.pt` files containing standard PyTorch Geometric keys `z` and `y` alongside NequIP keys. However, NequIP's batcher (`AtomicDataDict.batched_from_list`) enforces strict key verification and raises a `KeyError` when encountering any unregistered keys.
* **Resolution**: Filter out `z` and `y` from the dictionary before saving the `.pt` files, or use a custom loader that strips them out.

### ⚠️ 2. Inefficient Edge Computation
* **Impact**: **LOW (Medium at Scale)**
* **Details**: Pairwise distance thresholding is performed inside `preprocess.py` in Python, which scales as $O(N^2)$ and creates large files (since the edge index is stored explicitly).
* **Resolution**: Delegate neighbor list construction to NequIP's internal C++ backends at runtime using `NeighborListTransform`.

---

## 🛠️ Actionable Code Changes (For 10/10 Readiness)

To make the preprocessing pipeline fully compatible with NequIP without breaking other components, we propose updating `build_graph_data` in `scripts/preprocess.py` to only output NequIP-compliant keys when building the raw dictionary representation.

### Proposed Code Diff for `scripts/preprocess.py`:

```diff
 def build_graph_data(z, pos, energy, force, cutoff):
     """Construct PyTorch-ready dict representation of the graph with NequIP fields."""
     edge_index = compute_edge_index(pos, cutoff)
     
     # Wrap in PyTorch tensors
     data = {
-        'z': torch.tensor(z, dtype=torch.long),
         'atomic_numbers': torch.tensor(z, dtype=torch.long),
         'pos': torch.tensor(pos, dtype=torch.float32),
         'edge_index': torch.tensor(edge_index, dtype=torch.long),
-        'y': torch.tensor([energy], dtype=torch.float32),
         'total_energy': torch.tensor([energy], dtype=torch.float32),
         'forces': torch.tensor(force, dtype=torch.float32),
         'pbc': torch.tensor([False, False, False], dtype=torch.bool),
         'cell': torch.zeros((3, 3), dtype=torch.float32)
     }
     
     # Try wrapping in PyTorch Geometric Data if available
     try:
         from torch_geometric.data import Data
         data = Data(
-            x=data['z'].unsqueeze(-1), # optional standard node features
-            z=data['z'],
-            atomic_numbers=data['atomic_numbers'],
+            x=data['atomic_numbers'].unsqueeze(-1), # optional standard node features
+            atomic_numbers=data['atomic_numbers'],
             pos=data['pos'],
             edge_index=data['edge_index'],
-            y=data['y'],
-            total_energy=data['total_energy'],
+            total_energy=data['total_energy'],
             forces=data['forces'],
             pbc=data['pbc'],
             cell=data['cell']
         )
     except ImportError:
         pass
         
     return data
```

---

## 🚀 Future Roadmap

1. **Lazy Loading Database**: Convert large datasets into LMDB files using `NequIPLMDBDataset` to support training with millions of frames under a fixed memory footprint.
2. **Dynamic Neighbor List Construction**: Set `cutoff` to infinity in preprocessing to avoid storing edge indexes altogether, and instead use NequIP's runtime `NeighborListTransform` with optimized backends (`matscipy`/`vesin`).
3. **Chemical Symbol Mapping**: Ensure the training configuration incorporates `ChemicalSpeciesToAtomTypeMapper` to handle multi-element systems dynamically.
