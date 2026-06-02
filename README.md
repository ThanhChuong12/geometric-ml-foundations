# Geometric Machine Learning Foundations

A comprehensive exploration of Geometric Machine Learning, focusing on group theory, equivariant neural networks (such as NequIP), and weight space learning.

---

## 📖 Project Overview
This repository serves as a foundational platform for implementing and studying Geometric Machine Learning models. It includes:
* Implementations of $E(3)$-equivariant interatomic potential models.
* Standardized pipelines for preparing molecular dynamics trajectories.
* Investigations of symmetries, group representations, and physical constraints in neural networks.

---

## ⚙️ Setup Instructions

### Python Environment Setup
We utilize a virtual environment containing PyTorch, PyTorch Geometric, and the NequIP package. Refer to the [nequip-env/README.md](file:///d:/Regular_School/N3/HK2/ML/geometric-ml-foundations/nequip-env/README.md) for full instructions on setting up system dependencies and installing requirements:
```powershell
# Navigate to the environment folder
cd nequip-env

# Follow instructions in nequip-env/README.md to setup virtual environment and dependencies
```

---

## 📂 Project Structure

```
geometric-ml-foundations/
│
├── archive/                  # Historical reports and document archives
├── data/                     # Data directory (ignored on Git except mock structures)
│   ├── raw/                  # Raw trajectory files (.npz)
│   └── processed/            # Serialized graph datasets (.pt, .csv)
│
├── docs/                     # Technical documentation and audits
│   ├── preprocessing_audit.md
│   └── preprocessing_cleanup_plan.md
│
├── nequip-env/               # NequIP experimental virtual environment directory
│   ├── requirements.txt
│   └── README.md
│
├── report/                   # LaTeX source files and report materials (Exclusively LaTeX)
│
├── scripts/                  # Preprocessing and utility scripts
│   └── preprocess.py
│
└── README.md                 # Project entry documentation
```

---

## 🚀 Preprocessing Usage Summary

The preprocessing pipeline cleans Cartesian coordinate trajectories, centers positions at the center-of-mass, constructs neighborhood lists, shuffles/splits frames, and normalizes targets:
```powershell
python scripts/preprocess.py `
  --input_path data/raw/mock_md17.npz `
  --output_dir data/processed `
  --format pt `
  --cutoff 4.0 `
  --center `
  --normalize
```
For detailed arguments, requirements, and outputs, see the links below.

---

## 🔗 Detailed Documentation

* **Environment Configuration**: See [nequip-env/README.md](file:///d:/Regular_School/N3/HK2/ML/geometric-ml-foundations/nequip-env/README.md)
* **Preprocessing Pipeline Audit & Compatibility**: See [docs/preprocessing_audit.md](file:///d:/Regular_School/N3/HK2/ML/geometric-ml-foundations/docs/preprocessing_audit.md)
* **Documentation Cleanup Strategy**: See [docs/preprocessing_cleanup_plan.md](file:///d:/Regular_School/N3/HK2/ML/geometric-ml-foundations/docs/preprocessing_cleanup_plan.md)
