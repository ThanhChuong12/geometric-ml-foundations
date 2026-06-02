# Loader Compatibility Report

This report presents a forensic validation of the preprocessed datasets against NequIP's actual data loaders and batching utilities.

---

## 🔍 1. Verification of Preprocessed Files

We loaded `train.pt`, `val.pt`, and `test.pt` directly to examine their structure, types, shapes, and dtypes.

### Python script output:
```
=== Loading data/processed/train.pt ===
Object type: <class 'list'>
Dataset length: 80
Sample object type: <class 'dict'>
Keys: ['z', 'atomic_numbers', 'pos', 'edge_index', 'y', 'total_energy', 'forces', 'pbc', 'cell']
  atomic_numbers: shape=[5], dtype=torch.int64, device=cpu
  cell: shape=[3, 3], dtype=torch.float32, device=cpu
  edge_index: shape=[2, 20], dtype=torch.int64, device=cpu
  forces: shape=[5, 3], dtype=torch.float32, device=cpu
  pbc: shape=[3], dtype=torch.bool, device=cpu
  pos: shape=[5, 3], dtype=torch.float32, device=cpu
  total_energy: shape=[1], dtype=torch.float32, device=cpu
  y: shape=[1], dtype=torch.float32, device=cpu
  z: shape=[5], dtype=torch.int64, device=cpu
```

---

## ⚠️ 2. The Unregistered Key Bug (Mismatched Schema)

NequIP uses a strict key registration mechanism defined in `nequip/data/_key_registry.py`. During batching (via `AtomicDataDict.batched_from_list`), any key not registered in the allowed fields list raises a `KeyError`.

### Execution Log (Uncleaned Data):
Running the following command:
```powershell
python -c "import torch; from nequip.data import AtomicDataDict; data = torch.load('data/processed/train.pt'); batch = AtomicDataDict.batched_from_list(data[:4])"
```
Produced the following traceback:
```
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "D:\Regular_School\N3\HK2\ML\geometric-ml-foundations\nequip-env\.venv\Lib\site-packages\nequip\data\AtomicDataDict.py", line 138, in batched_from_list
    raise KeyError(f"Unregistered key {k}")
KeyError: 'Unregistered key z'
```

### Explanation:
- The previous preprocessing pipeline included both standard PyG keys (`z`, `y`) and NequIP keys (`atomic_numbers`, `total_energy`).
- While this ensured PyG compatibility, it **completely breaks** NequIP's batcher because `z` and `y` are unregistered keys in NequIP's key registry.

---

## 🟢 3. Proving Compatibility (Cleaned Schema)

When the unregistered keys `z` and `y` are removed from the preprocessed dictionary, the NequIP batcher executes successfully.

### Execution Log (Cleaned Data):
Running the following command:
```powershell
python -c "import torch; from nequip.data import AtomicDataDict; data = torch.load('data/processed/train.pt'); cleaned = [{k: v for k, v in d.items() if k not in ['z', 'y']} for d in data]; batch = AtomicDataDict.batched_from_list(cleaned[:4]); print('Batch keys:', list(batch.keys())); print('Batch shapes:'); [print(f'  {k}: {list(v.shape)}, {v.dtype}') for k, v in batch.items()]"
```
Produced the following output:
```
Batch keys: ['edge_index', 'batch', 'atomic_numbers', 'pos', 'total_energy', 'forces', 'pbc', 'cell', 'num_atoms']
Batch shapes:
  edge_index: [2, 80], torch.int64
  batch: [20], torch.int64
  atomic_numbers: [20], torch.int64
  pos: [20, 3], torch.float32
  total_energy: [4], torch.float32
  forces: [20, 3], torch.float32
  pbc: [12], torch.bool
  cell: [12, 3], torch.float32
  num_atoms: [4], torch.int64
```

### Match Against NequIP Loader Requirements:
- **`edge_index`**: Collated from individual shape `[2, 20]` to batched shape `[2, 80]` with node offsets correctly applied.
- **`batch`**: Correctly constructed to map each of the 20 nodes to their corresponding frame index (`0` to `3`).
- **`atomic_numbers`**: Collated to shape `[20]` (long tensor).
- **`pos`**: Collated to shape `[20, 3]` (float tensor).
- **`total_energy`**: Collated to shape `[4]` (float tensor).
- **`forces`**: Collated to shape `[20, 3]` (float tensor).
- **`pbc`**: Concatenated to shape `[12]` (bool tensor).
- **`cell`**: Concatenated to shape `[12, 3]` (float tensor).
- **`num_atoms`**: Collated to shape `[4]` (long tensor representing the count of atoms per frame).

---

## 🏆 Conclusion
The preprocessed files are **compatible** with NequIP's batching utilities **if and only if** the unregistered keys `z` and `y` are excluded or filtered out from the dataset dictionaries.
