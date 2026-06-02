# NequIP — Môi trường thực nghiệm Paper 2

Thư mục này chứa môi trường Python độc lập để chạy và nghiên cứu mô hình
**NequIP** (Neural Equivariant Interatomic Potentials), Paper 2 trong đồ án
Geometric Machine Learning.

> Batzner et al., "E(3)-equivariant graph neural networks for data-efficient
> and accurate interatomic potentials", *Nature Communications*, 2022.
> https://doi.org/10.1038/s41467-022-29939-5

---

## Yêu cầu hệ thống

| Thành phần | Phiên bản |
|---|---|
| Python | 3.13.x |
| CUDA | 12.2 trở lên |
| GPU | NVIDIA (khuyến nghị, có thể chạy CPU) |
| OS | Windows 10/11 |

---

## Cấu trúc thư mục

```
nequip-env/
├── .venv/              <- Môi trường Python ảo (không commit lên Git)
└── requirements.txt    <- Danh sách toàn bộ packages đã cài
```

---

## Hướng dẫn cài đặt

### Bước 1: Tạo virtual environment

Mở terminal tại thư mục gốc `geometric-ml-foundations/`, chạy:

```powershell
py -3.13 -m venv nequip-env\.venv
```

### Bước 2: Kích hoạt môi trường ảo

```powershell
# Windows PowerShell
& "nequip-env\.venv\Scripts\Activate.ps1"
```

Sau khi kích hoạt, terminal sẽ hiện prefix `(.venv)` ở đầu dòng:

```
(.venv) PS D:\TU HOC\geometric-ml-foundations>
```

### Bước 3: Cài PyTorch với CUDA

```powershell
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

### Bước 4: Cài toàn bộ dependencies từ requirements.txt

```powershell
pip install -r nequip-env\requirements.txt
```

### Bước 5: Cài NequIP trực tiếp từ GitHub

```powershell
pip install git+https://github.com/mir-group/nequip.git
```

---

## Kiểm tra cài đặt

Sau khi cài xong, kiểm tra bằng lệnh sau (trong môi trường đã activate):

```powershell
# Kiểm tra PyTorch và CUDA
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# Kiểm tra NequIP
python -c "import nequip; print('NequIP OK')"

# Kiểm tra e3nn (thư viện equivariant)
python -c "import e3nn; print('e3nn:', e3nn.__version__)"
```

Kết quả mong đợi:

```
PyTorch: 2.6.0+cu124
CUDA available: True
NequIP OK
e3nn: 0.6.0
```

---

## Các package chính và vai trò

| Package | Version | Vai trò |
|---|---|---|
| `torch` | 2.6.0+cu124 | Deep learning framework, chạy GPU |
| `nequip` | git@main | Mô hình NequIP chính |
| `e3nn` | 0.6.0 | Equivariant neural networks (E(3) symmetry) |
| `torch-geometric` | 2.7.0 | Graph neural network framework |
| `ase` | 3.28.0 | Atomistic Simulation Environment, đọc dataset phân tử |
| `wandb` | 0.27.0 | Theo dõi thí nghiệm và visualize kết quả |
| `hydra-core` | 1.3.2 | Quản lý config file |
| `pytorch-lightning` | 2.6.5 | Training loop có cấu trúc |
| `numpy` | 2.4.6 | Tính toán số học |
| `matplotlib` | 3.10.9 | Vẽ đồ thị kết quả |
| `scipy` | 1.17.1 | Tính toán khoa học |

---

## Cách dùng nhanh

### Chạy demo inference với pretrained model

```powershell
# Activate môi trường trước
& "nequip-env\.venv\Scripts\Activate.ps1"

# Clone repo NequIP để lấy config mẫu
git clone https://github.com/mir-group/nequip.git
cd nequip

# Chạy training thử trên dataset MD-17
python scripts/train.py configs/example.yaml
```

### Tắt môi trường ảo khi xong

```powershell
deactivate
```

---

## Lưu ý

- Thư mục `.venv/` **không nên commit lên Git** vì rất nặng (hàng trăm MB).
  File `.gitignore` đã được cấu hình để bỏ qua thư mục này.
- Chỉ cần commit `requirements.txt` để người khác có thể tái tạo môi trường.
- Nếu chạy trên máy không có GPU, PyTorch tự động fallback sang CPU,
  nhưng sẽ rất chậm với NequIP.
