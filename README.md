# Geometric Machine Learning — Các Bước Tiến Mới

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-16.2-000000?style=for-the-badge&logo=next.js&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.11x-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![NequIP](https://img.shields.io/badge/NequIP-E(3)--Equivariant-blueviolet?style=for-the-badge)

![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

</div>

> **Báo cáo Đồ án 3 — Nhập môn học máy (Introduction to Machine Learning)**
>
> *Khoa Công nghệ Thông tin, Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM*

---

## Table of Contents

- [1. About The Project](#1-about-the-project)
- [2. Theoretical Coverage](#2-theoretical-coverage)
  - [Part 1: Theory Report](#part-1-theory-report)
  - [Part 2: Experiments](#part-2-experiments)
  - [Part 3: Recent Research](#part-3-recent-research)
- [3. Models Implemented](#3-models-implemented)
- [4. Repository Structure](#4-repository-structure)
- [5. Getting Started](#5-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Demo Website](#running-the-demo-website)
  - [Running AI Training Experiments](#running-ai-training-experiments)
  - [Running the Notebooks](#running-the-notebooks)
- [6. Contributors](#6-contributors)
- [7. License & Acknowledgments](#7-license--acknowledgments)

---

## 1. About The Project

This project is **Assignment Report No. 3** for the *Introduction to Machine Learning* course at VNU-HCM University of Science. It provides a comprehensive study of **Geometric Machine Learning** — the mathematical framework unifying modern deep learning architectures under the lens of symmetry, invariance, and equivariance.

The project is organized into three interconnected components:

| **Part 1: Theory Report** | **Part 2: Experiments** | **Part 3: Recent Research** |
| :--- | :--- | :--- |
| Full Vietnamese re-exposition of Geometric Deep Learning foundations, covering group theory, equivariance, invariance, and the 5G framework. | Three end-to-end experiments (MNIST rotation invariance, PointNet 3D classification, NequIP molecular energy prediction) with interactive web demo. | In-depth survey of 6 cutting-edge papers (2022–2026) on parameter space symmetry, weight space learning, scaling laws, and model merging. |
| Covers symmetry groups $SO(2)$, $SO(3)$, $SE(3)$, $E(3)$, group actions, equivariant architectures (NequIP, PointNet), Frame Averaging, Data Augmentation. | Full training pipelines for all three models; ablation studies (baseline $L_0$ vs equivariant $L_1$, 100 vs 1000 samples) on QM9 dataset. | Each paper analyzed along four axes: Motivation, Contribution, Theory Connection, and Limitation. Includes reproducible experiment scripts. |
| Every theorem/definition has concrete examples; key concepts illustrated with TikZ diagrams. | All results reproducible with `seed=42`; model weights included in `storage/weights/`. | Covers NTK theory, Weight Space Learning, Git Re-Basin merging, group averaging approximation, symmetry testing, and scaling laws. |

> **Core Philosophy:** Each model is trained and deployed end-to-end. The demo website lets anyone experiment with rotation-invariant digit recognition, 3D point cloud classification, and E(3)-equivariant molecular energy prediction — all running live in the browser against real trained models.

---

## 2. Theoretical Coverage

### Part 1: Theory Report

The LaTeX report (written in Vietnamese) covers the theoretical foundations of Geometric Deep Learning, anchored in the blueprint by Bronstein et al. (2021) and extended by Zhao et al. (2026), organized across 7 chapters.

| Report Chapter | Content |
| :--- | :--- |
| **Chương 1 — Giới thiệu** | Motivation: architecture fragmentation, curse of dimensionality; the geometric prior as a remedy; parameter space symmetry and weight space learning as the new frontier |
| **Chương 2 — Kiến thức nền tảng** | Group theory fundamentals: permutation groups $S_n$, cyclic groups $C_n$, rotation groups $SO(d)$, Euclidean groups $SE(d)$, $E(3)$; group actions, orbits; invariance vs. equivariance with commutative diagrams |
| **Chương 3 — Mô hình bảo toàn đối xứng** | Two approaches: (1) data-level intervention — Data Augmentation, Group Averaging (Reynolds operator), Frame Averaging; (2) architecture-level design — equivariant convolutions, PointNet (max-pool invariance + T-Net canonicalization), NequIP (E(3)-equivariant message passing on atomic graphs); robotics applications |
| **Chương 4 — Lý thuyết nâng cao & hướng mới** | Sample complexity benefits of symmetry; Expanders for scalable group averaging; symmetry testing in data; parameter space symmetry (neuron permutation, ReLU scale); loss landscape connectivity; Git Re-Basin merging; weight space learning with equivariant meta-networks; flexible/learned symmetries; geometry beyond symmetry |
| **Chương 5 — Thực nghiệm** | Data efficiency & scaling laws (NequIP vs. baseline on QM9/rMD17/OC20); group averaging approximation with 32-sample subset vs. full $10^6$-element group; weight space learning with LoRA meta-networks ($R^2 = 0.999$); loss landscape & Git Re-Basin merging; symmetry testing on QM9 (canonicalized coordinate observation) |
| **Chương 6 — Đánh giá & kết luận** | Tradeoff analysis: when to use equivariant models; limitations; future directions |
| **Chương 7 — Demo minh họa** | Detailed write-up of all three demo experiments linking experimental results back to theory |

**Key theorems and concepts covered with full proofs and examples:**

- **Invariance & Equivariance** — Formal commutative diagram definition; examples: translation equivariance in CNN, permutation equivariance in PointNet, $E(3)$-equivariance in NequIP
- **Group Averaging (Reynolds Operator)** — $\langle\Phi\rangle_G(x) = \frac{1}{|G|}\sum_{g\in G} \rho(g)\Phi(g^{-1}\cdot x)$ is always $G$-invariant; approximation via random 32-element subset
- **Frame Averaging** — $\langle\Phi\rangle_\mathcal{F}(x) = \frac{1}{|\mathcal{F}(x)|}\sum_{f\in\mathcal{F}(x)}\Phi(f\cdot x)$ with data-dependent frame; $C_4$-invariance for MNIST rotation demo
- **PointNet Max-Pool Invariance** — Global max pooling is permutation-invariant; T-Net learns $3\times3$ alignment transform; critical point set theorem (Theorem 2 in Qi et al. 2017)
- **NequIP $E(3)$-Equivariance** — Message passing with spherical harmonic edge features; energy is $E(3)$-invariant scalar; forces are $E(3)$-equivariant vectors
- **Parameter Space Symmetry** — Neuron permutation symmetry; ReLU positive-scaling symmetry; Mode connectivity and loss landscape topology; Git Re-Basin permutation alignment

### Part 2: Experiments

Three experiments, each implemented as a training pipeline (`ai_core/`) and an interactive demo (`frontend/` + `backend/`).

| Experiment | Task | Models Compared | Dataset | Key Finding |
| :--- | :--- | :--- | :--- | :--- |
| **Part 1 — MNIST Rotation Invariance** | Handwritten digit recognition under arbitrary rotation | Baseline CNN / Data Augmentation / Frame Averaging ($C_4$) | MNIST | Frame Averaging maintains stable accuracy across all rotation angles; Baseline CNN degrades sharply beyond ±30° |
| **Part 2 — PointNet 3D Classification** | Point cloud object classification | PointNet Basic (no T-Net) / PointNet Full (T-Net + regularization) | ModelNet-40 subset (5 classes: airplane, car, chair, lamp, table) | Full model with T-Net is significantly more robust to input rotation and point dropout; critical point visualization shows sparse skeleton is sufficient |
| **Part 3 — NequIP Molecular Energy Prediction** | Quantum mechanical energy of small molecules | Baseline $L_0$ (non-equivariant MLP) vs NequIP $L_1$ ($E(3)$-equivariant GNN) | QM9 subset (100 and 1000 samples) | NequIP reaches the same accuracy as baseline with **1000× fewer training samples**; equivariant model scales dramatically better |

**Ablation configurations trained:**

| Config | Model | Training samples | Notes |
| :---: | :--- | :---: | :--- |
| `baseline_l0_100` | Non-equivariant MLP | 100 | Reference: no geometric prior |
| `baseline_l0_1000` | Non-equivariant MLP | 1000 | Larger data baseline |
| `nequip_l1_100` | NequIP $L_{\max}=1$ | 100 | Equivariant; data-scarce regime |
| `nequip_l1_1000` | NequIP $L_{\max}=1$ | 1000 | Equivariant; full training set |

All checkpoints are saved under `ai_core/outputs/<config>/best.ckpt`.

### Part 3: Recent Research

Survey of 6 papers (2022–2026) from top venues. Each analyzed along four axes: **Motivation, Contribution, Connection to Theory, and Limitation**.

| Paper | Venue | Topic | Theory Connection |
| :--- | :---: | :--- | :--- |
| *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges* (Bronstein et al.) | arXiv 2021 | 5G framework unifying CNN/RNN/GNN/Transformer | Foundation: equivariant blueprint for all architectures |
| *Symmetry in Neural Network Parameter Spaces* (Zhao et al.) | arXiv 2026 | Parameter space symmetry; weight space learning | Shifts GDL focus from data symmetry to model symmetry |
| *Git Re-Basin: Merging Models modulo Permutation Symmetries* (Ainsworth et al.) | ICLR 2023 | Permutation alignment for loss-barrier-free model merging | Direct application of neuron permutation symmetry |
| *Scaling Laws for Equivariant Representations* (Brehmer et al.) | arXiv 2025 | Data efficiency & compute scaling of equivariant models | Geometric priors improve scaling exponent |
| *Group Average Approximation via Expanders* (Tahmasebi et al.) | arXiv 2025 | Scalable group averaging with $O(1)$ subset | Makes Reynolds operator tractable for large groups |
| *Symmetry Testing in Machine Learning* (Soleymani et al.) | arXiv 2025 | Statistical hypothesis testing for data symmetry | NP-hard full test → tractable random sampling approximation |

---

## 3. Models Implemented

### MNIST — Rotation Invariance (Part 1)

| Model | Architecture | Training Strategy | Weights |
| :--- | :--- | :--- | :--- |
| **Baseline CNN** | Conv2d + MaxPool + FC | Standard; upright digits only | `storage/weights/model_baseline.pth` |
| **Data Augmentation CNN** | Same architecture | Random rotation augmentation during training | `storage/weights/model_augmented.pth` |
| **Frame Averaging CNN** | Same architecture | Upright training; $C_4$-frame averaging at inference (avg. over 0°, 90°, 180°, 270°) | `storage/weights/model_frame.pth` |

### PointNet — 3D Point Cloud Classification (Part 2)

| Model | Architecture | Key Components | Weights |
| :--- | :--- | :--- | :--- |
| **PointNet Basic** | Shared MLP + Global Max Pool | No T-Net; permutation-invariant by max pool | `storage/weights/pointnet_basic.pth` |
| **PointNet Full** | Shared MLP + Global Max Pool | Input T-Net (3×3) + Feature T-Net (64×64) + orthogonality regularization loss | `storage/weights/pointnet_cls.pth` |

Both models classify 5 classes: `airplane`, `car`, `chair`, `lamp`, `table`.

### NequIP — E(3)-Equivariant Molecular Energy (Part 3)

| Config | Architecture | Equivariance | Weights |
| :--- | :--- | :--- | :--- |
| **Baseline $L_0$** | MLP on scalar features | None ($L_{\max}=0$) | `ai_core/outputs/baseline_l0_1000/best.ckpt` |
| **NequIP $L_1$** | E(3)-equivariant GNN with spherical harmonic edge embeddings ($L_{\max}=1$) | Full $E(3)$ | `ai_core/outputs/nequip_l1_1000/best.ckpt` |

The production server loads `nequip_l1_1000/best.ckpt` for the Part 3 demo.

---

## 4. Repository Structure

```text
geometric-ml-foundations/
│
├── ai_core/                          # ML training & research code (Python)
│   ├── configs/                      # Hydra YAML configs for 4 ablation experiments
│   │   ├── baseline_l0_100.yaml      # Non-equivariant MLP, 100 samples
│   │   ├── baseline_l0_1000.yaml     # Non-equivariant MLP, 1000 samples
│   │   ├── nequip_l1_100.yaml        # NequIP L1, 100 samples
│   │   └── nequip_l1_1000.yaml       # NequIP L1, 1000 samples
│   │
│   ├── outputs/                      # Training results & model checkpoints
│   │   ├── baseline_l0_100/          # best.ckpt, last.ckpt, metrics.csv
│   │   ├── baseline_l0_1000/
│   │   ├── nequip_l1_100/
│   │   └── nequip_l1_1000/
│   │
│   ├── scripts/
│   │   ├── prepare_qm9_subsets.py    # Download & split QM9 into 100/1000 subsets
│   │   ├── preprocess.py             # Build NequIP-compatible graph tensors
│   │   ├── train_full_v2.py          # PointNet training script (with augmentation)
│   │   ├── train_pointnet.py         # PointNet Basic training script
│   │   ├── download_assets.py        # Download sample point clouds & model weights
│   │   └── run_all_experiments.sh    # One-shot: run all 4 NequIP ablations sequentially
│   │
│   ├── notebooks/
│   │   ├── 01_mnist_rotation_invariance.ipynb    # MNIST training + rotation analysis
│   │   ├── 02_modelnet_pointcloud_classification.ipynb  # PointNet training + critical pts
│   │   ├── 03_qm9_molecular_energy_prediction.ipynb     # NequIP ablation + learning curves
│   │   ├── nequip_learning_curve.py              # Learning curve plotting utilities
│   │   ├── pointnet_utils.py                     # Shared PointNet helpers
│   │   └── figures/                              # Generated plots (PNG)
│   │
│   ├── evaluate.py                   # Unified evaluation script for all checkpoints
│   └── requirements.txt              # (empty — see nequip-env/requirements.txt)
│
├── backend/                          # FastAPI inference server (Python)
│   ├── main.py                       # App entry point; lifespan startup (loads NequIP)
│   ├── requirements.txt              # Backend Python dependencies
│   │
│   ├── api/
│   │   ├── part1_routes.py           # POST /api/part1/predict — MNIST digit
│   │   ├── part2_routes.py           # POST /api/part2/classify — PointNet 3D
│   │   └── part3_routes.py           # POST /api/part3/predict-energy — NequIP
│   │
│   ├── core/
│   │   ├── frame_averaging.py        # C4-frame averaging inference logic
│   │   ├── nequip_wrapper.py         # NequIPPredictor: load checkpoint, run inference
│   │   └── pointnet_model.py         # PointNet Basic & Full model definitions (PyTorch)
│   │
│   ├── services/
│   │   ├── model_service.py          # MNIST prediction (baseline / augmented / frame)
│   │   ├── pointnet_service.py       # PointNet inference + critical point extraction
│   │   └── qm9_service.py            # Molecular graph construction for NequIP
│   │
│   ├── models/
│   │   ├── schemas.py                # MNIST Pydantic request/response models
│   │   └── pointnet_schemas.py       # PointNet Pydantic schemas
│   │
│   └── schemas/
│       └── prediction.py             # NequIP MoleculeInput / EnergyResponse schemas
│
├── frontend/                         # Next.js 16 interactive demo (TypeScript/React)
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx              # Main page: 3-tab layout (Part 1 / 2 / 3)
│   │   │   ├── layout.tsx            # Root layout
│   │   │   └── globals.css           # Tailwind CSS base styles
│   │   │
│   │   ├── components/
│   │   │   ├── DigitCanvas.tsx       # HTML5 Canvas drawing board for MNIST
│   │   │   ├── RotationSlider.tsx    # Rotation angle slider (Part 1)
│   │   │   ├── PredictionCard.tsx    # Model comparison card (Part 1)
│   │   │   ├── PointCloudViewer.tsx  # Three.js 3D point cloud viewer (Part 2)
│   │   │   ├── PointNetResultCard.tsx # PointNet result comparison (Part 2)
│   │   │   ├── PerturbationControls.tsx # Rotation + noise + dropout sliders (Part 2)
│   │   │   ├── MoleculeViewer3D.tsx  # Three.js molecular structure viewer (Part 3)
│   │   │   ├── MoleculeEditor.tsx    # Atom position editor table (Part 3)
│   │   │   ├── ExampleMolecules.tsx  # Preset molecules selector (Part 3)
│   │   │   ├── EnergyResultCard.tsx  # NequIP energy result display (Part 3)
│   │   │   ├── MathTerm.tsx          # Inline math term with tooltip definition
│   │   │   ├── UsageGuide.tsx        # Per-tab usage instructions
│   │   │   └── HeaderDecorator.tsx   # Decorative header SVG
│   │   │
│   │   ├── services/
│   │   │   ├── predictionService.ts  # MNIST API client
│   │   │   ├── pointnetService.ts    # PointNet API client
│   │   │   └── nequipService.ts      # NequIP API client
│   │   │
│   │   └── types/
│   │       ├── index.ts              # MNIST types
│   │       ├── pointnet.ts           # PointNet types
│   │       └── molecule.ts           # Atom / molecule types
│   │
│   ├── package.json
│   └── next.config.ts
│
├── storage/                          # Model weights & datasets
│   ├── weights/
│   │   ├── model_baseline.pth        # MNIST Baseline CNN
│   │   ├── model_augmented.pth       # MNIST Data Augmentation CNN
│   │   ├── model_frame.pth           # MNIST Frame Averaging CNN
│   │   ├── pointnet_basic.pth        # PointNet Basic
│   │   └── pointnet_cls.pth          # PointNet Full (with T-Net)
│   └── data/
│       ├── MNIST/                    # MNIST dataset (auto-downloaded)
│       ├── qm9/                      # QM9 subsets (100 / 1000 samples, .extxyz)
│       └── sample_clouds/            # Pre-sampled ModelNet point clouds (.npy)
│
├── report/
│   ├── src/
│   │   ├── main.tex                  # LaTeX root file (hcmus-report class)
│   │   ├── hcmus-report.cls          # HCMUS report document class
│   │   ├── codespace.sty             # Custom styling package
│   │   ├── chapters/                 # 7 chapter .tex files
│   │   │   ├── 01_gioi_thieu.tex
│   │   │   ├── 02_kien_thuc_chuan_bi.tex
│   │   │   ├── 03_mo_hinh_equivariant.tex
│   │   │   ├── 04_data_aug_va_huong_moi.tex
│   │   │   ├── 05_thuc_nghiem.tex
│   │   │   ├── 06_danh_gia_ket_luan.tex
│   │   │   └── 07_demo.tex
│   │   ├── graphics/                 # Figures & TikZ diagrams
│   │   └── refs/
│   │       └── references.bib        # BibTeX references
│   ├── output/                       # Compiled PDF
│   └── README.md                     # Report build instructions
│
├── nequip-env/
│   └── requirements.txt              # Pinned NequIP environment (torch, nequip, e3nn...)
│
├── .gitignore
├── LICENSE
└── README.md
```

---

## 5. Getting Started

### Prerequisites

| Tool | Version | Purpose |
| :--- | :--- | :--- |
| **Python** | 3.10+ | Backend, AI training |
| **Node.js** | 18+ | Frontend |
| **CUDA** (optional) | 11.8+ | GPU training; CPU fallback is automatic |
| **Git** | any | Clone repo |

### Installation

**Step 1: Clone the repository**

```bash
git clone https://github.com/ThanhChuong12/geometric-ml-foundations.git
cd geometric-ml-foundations
```

**Step 2: Set up the Python environment for the NequIP / AI core**

Create a virtual environment and install the pinned dependencies:

```bash
python -m venv nequip-env
# Windows
nequip-env\Scripts\activate
# Linux / macOS
source nequip-env/bin/activate

pip install -r nequip-env/requirements.txt
```

> The `requirements.txt` pins every dependency (torch 2.6, nequip, e3nn 0.6, pytorch-lightning 2.6, etc.) for full reproducibility.

**Step 3: Install backend dependencies**

```bash
pip install -r backend/requirements.txt
```

**Step 4: Install frontend dependencies**

```bash
cd frontend
npm install
cd ..
```

---

### Running the Demo Website

The demo is a three-tab web application. You need to start the **backend** and **frontend** separately.

#### Start the Backend (FastAPI)

From the project root, with the Python environment activated:

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

On startup, the server will:
1. Load the NequIP checkpoint from `ai_core/outputs/nequip_l1_1000/best.ckpt`
2. Cache the `MoleculeGraphService` with learned `r_max` and atom type names
3. Register three API route groups: `/api/part1`, `/api/part2`, `/api/part3`

You can verify the server is healthy at: `http://localhost:8000/health`

> **Note:** The first startup takes ~30–60 seconds to deserialize the NequIP checkpoint. Subsequent requests are fast.

#### Start the Frontend (Next.js)

In a separate terminal:

```bash
cd frontend
npm run dev
```

Open your browser at **`http://localhost:3000`**

#### Demo Tabs

| Tab | Endpoint | What to do |
| :--- | :--- | :--- |
| **Part 1 — MNIST (SO(2))** | `POST /api/part1/predict` | Draw a digit on the canvas → adjust the rotation slider → click **Chạy Phân loại** → compare Baseline / Data Aug / Frame Averaging results side-by-side |
| **Part 2 — PointNet 3D** | `POST /api/part2/classify` | Select an object class (airplane, car, chair, lamp, table) → drag to rotate the point cloud → add noise/dropout perturbations → click **Chạy Phân Loại** → toggle **Critical Points** mode to see the sparse decision skeleton |
| **Part 3 — NequIP Energy** | `POST /api/part3/predict-energy` | Pick a preset molecule (H₂O, CH₄, NH₃, CO₂, C₂H₄...) or edit atomic coordinates manually → click **Dự đoán Năng lượng** → observe the predicted quantum mechanical energy in eV |

#### Download Sample Assets (Part 2)

If `storage/data/sample_clouds/` is empty, run the asset downloader first:

```bash
python ai_core/scripts/download_assets.py
```

This fetches the pre-sampled ModelNet-40 `.npy` point clouds for all 5 demo classes.

---

### Running AI Training Experiments

#### Part 1 — MNIST (Jupyter Notebook)

```bash
jupyter notebook ai_core/notebooks/01_mnist_rotation_invariance.ipynb
```

This notebook trains all three MNIST models (Baseline / Augmented / Frame Averaging), evaluates rotation robustness, and saves weights to `storage/weights/`.

#### Part 2 — PointNet (Script or Notebook)

```bash
# PointNet Full (with T-Net, recommended)
python ai_core/scripts/train_full_v2.py

# Or interactively:
jupyter notebook ai_core/notebooks/02_modelnet_pointcloud_classification.ipynb
```

Weights are saved to `storage/weights/pointnet_cls.pth`.

#### Part 3 — NequIP Ablation Study

**Step 1: Prepare QM9 subsets** (requires downloading the raw QM9 dataset first)

```bash
python ai_core/scripts/prepare_qm9_subsets.py \
  --root storage/data/qm9_raw \
  --output-dir storage/data/qm9 \
  --seed 42 \
  --subset-sizes 100 1000
```

**Step 2: Run all 4 ablation experiments**

```bash
bash ai_core/scripts/run_all_experiments.sh
```

This runs `nequip-train` sequentially for all four configs (`baseline_l0_100`, `baseline_l0_1000`, `nequip_l1_100`, `nequip_l1_1000`) and saves checkpoints to `ai_core/outputs/<config>/`.

Or run a single config manually:

```bash
nequip-train \
  --config-path ai_core/configs \
  --config-name nequip_l1_1000 \
  hydra.run.dir=ai_core/outputs/nequip_l1_1000
```

**Step 3: Evaluate and plot learning curves**

```bash
python ai_core/evaluate.py

# Generate learning curve plots
python ai_core/notebooks/nequip_learning_curve.py \
  --results-dir ai_core/outputs \
  --output-dir ai_core/notebooks/figures
```

Or open the full analysis notebook:

```bash
jupyter notebook ai_core/notebooks/03_qm9_molecular_energy_prediction.ipynb
```

---

### Running the Notebooks

All three notebooks are standalone — each manages its own data loading and seeding.

```bash
# Recommended order:
jupyter notebook ai_core/notebooks/01_mnist_rotation_invariance.ipynb
jupyter notebook ai_core/notebooks/02_modelnet_pointcloud_classification.ipynb
jupyter notebook ai_core/notebooks/03_qm9_molecular_energy_prediction.ipynb
```

---

## 6. Contributors

This project was developed by a team of 5 students from the *Faculty of Information Technology, VNU-HCM University of Science*.

| Contributor | Student ID | Role | Main Contributions |
| :--- | :---: | :--- | :--- |
| **Lê Hà Thanh Chương** | `23120195` | **Project Lead** | Repo architecture & monorepo setup; FastAPI backend (NequIP inference pipeline, CPU-only deployment); frontend NequIP tab (MoleculeViewer3D, MoleculeEditor, preset molecules, EnergyResultCard); NequIP training configs & QM9 subset pipeline; report structure & BibTeX QA; PR reviews and final integration |
| **Trà Văn Sỹ** | `23120197` | **ML Engineer** | MNIST demo frontend (Part 1 tab: DigitCanvas, RotationSlider, PredictionCard); report chapter on data augmentation (Chương 4); research survey on scaling laws and Git Re-Basin; LaTeX formatting |
| **Huỳnh Đức Thịnh** | `23120199` | **Full-stack Engineer** | Frontend UI polish (header, tab bar, animations, molecule presets); 3D viewer rotation sync; PointNet notebook; report chapters on equivariant model architectures (Chương 3) and NTK theory |
| **Bùi Trung Hiếu** | `23120257` | **ML Researcher** | NequIP training notebooks (learning curves, 3D molecule visualization, training/validation loss plots, bar chart metrics); log parsing utilities; QM9 visualization (7-molecule 3D plots); report sections on NequIP and molecular simulation |
| **Lê Công Phúc** | `23120330` | **ML Engineer** | PointNet full-stack (Part 2 backend, PointNet Basic & Full model implementations, classification service, critical point extraction); preprocessing pipeline (`preprocess.py`); edge index stability fixes; automated testing; report sections on PointNet and theoretical proofs |

---

## 7. License & Acknowledgments

### Academic Acknowledgments

This project is **Assignment Report No. 3** for the *Introduction to Machine Learning* course at *VNU-HCM University of Science*.

The team sincerely thanks:

- **Instructor:** ThS. Lê Nhựt Nam (lnnam@fit.hcmus.edu.vn)

### Paper & Framework Attribution

- **Geometric Deep Learning Blueprint:** *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges* — Bronstein, Bruna, Cohen, Veličković. arXiv 2021. ([arXiv](https://arxiv.org/abs/2104.13478))

- **NequIP:** *E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials* — Batzner et al. Nature Communications, 2022. ([doi](https://doi.org/10.1038/s41467-022-29939-5))

- **PointNet:** *PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation* — Qi et al. CVPR 2017. ([arXiv](https://arxiv.org/abs/1612.00593))

- **Git Re-Basin:** *Git Re-Basin: Merging Models modulo Permutation Symmetries* — Ainsworth et al. ICLR 2023. ([arXiv](https://arxiv.org/abs/2209.04836))

- **Parameter Space Symmetry:** *Symmetry in Neural Network Parameter Spaces* — Zhao et al. arXiv 2026.

- **Scaling Laws:** *Scaling Laws for Equivariant Representations* — Brehmer et al. arXiv 2025.

- **Group Averaging Approximation:** *On the Approximation of Group Averaging* — Tahmasebi et al. arXiv 2025.

- **Symmetry Testing:** *Symmetry Testing in Machine Learning* — Soleymani et al. arXiv 2025.

- **QM9 Dataset:** *Quantum chemistry structures and properties of 134 kilo molecules* — Ramakrishnan et al. Scientific Data, 2014.

- **Frameworks:** [NequIP](https://github.com/mir-group/nequip), [e3nn](https://e3nn.org/), [PyTorch Geometric](https://pyg.org/), [FastAPI](https://fastapi.tiangolo.com/), [Next.js](https://nextjs.org/), [Three.js](https://threejs.org/), [React Three Fiber](https://docs.pmnd.rs/react-three-fiber)

### License

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

The source code of this project is distributed under the **MIT License**.
See the `LICENSE` file for full details.

<br>
<p align="center">
  <i>Built by the ML Team | University of Science, VNU-HCM | 2026</i>
</p>
