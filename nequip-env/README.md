# Môi trường Thực nghiệm & Backend Demo (Part 2 & Part 3)

Thư mục này chứa cấu hình và môi trường Python ảo độc lập (`.venv`) đồng nhất cho dự án **Geometric Machine Learning**, được thiết kế để chạy:
1. **Huấn luyện & Thực nghiệm (`ai_core`):** Chạy mã nguồn huấn luyện, tiền xử lý dữ liệu và các Jupyter Notebook của **Part 2** (PointNet trên ModelNet40) và **Part 3** (NequIP trên QM9).
2. **Inference Backend (`backend`):** Phục vụ server FastAPI cung cấp các API dự đoán cho cả 3 phần của Demo Web (Part 1: MNIST, Part 2: PointNet, Part 3: NequIP).

---

## Yêu cầu hệ thống

| Thành phần | Phiên bản | Ghi chú |
|---|---|---|
| **Python** | 3.13.x | Khuyên dùng Python 3.13 |
| **CUDA** | 12.2 trở lên | Cần thiết nếu muốn huấn luyện mô hình (GPU NVIDIA) |
| **GPU** | NVIDIA | Khuyến nghị cho training; backend demo chạy mặc định trên CPU |
| **OS** | Windows 10/11 | Hoặc Linux (WSL) |

---

## Cấu trúc thư mục

```text
nequip-env/
├── .venv/              <- Môi trường Python ảo (đã được cấu hình .gitignore)
└── requirements.txt    <- Danh sách toàn bộ thư viện cần thiết cho cả Core AI và Backend
```

---

## Hướng dẫn cài đặt

### Bước 1: Tạo virtual environment
Mở terminal tại thư mục gốc của repository (`geometric-ml-foundations/`), chạy lệnh sau để tạo môi trường ảo đặt trong thư mục `nequip-env`:
```powershell
py -3.13 -m venv nequip-env\.venv
```

### Bước 2: Kích hoạt môi trường ảo
```powershell
# Windows PowerShell
& "nequip-env\.venv\Scripts\Activate.ps1"
```
Sau khi kích hoạt thành công, terminal sẽ hiển thị tiền tố `(.venv)` ở đầu dòng lệnh.

### Bước 3: Cài đặt PyTorch (hỗ trợ CUDA)
Huấn luyện PointNet và đặc biệt là NequIP yêu cầu GPU để đạt hiệu năng tốt nhất. Tiến hành cài đặt PyTorch hỗ trợ CUDA 12.4:
```powershell
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```
*(Nếu chỉ chạy demo backend trên CPU, bạn có thể chạy `pip install torch==2.6.0` mà không cần CUDA index).*

### Bước 4: Cài đặt các thư viện từ requirements.txt
```powershell
pip install -r nequip-env\requirements.txt
```

### Bước 5: Cài đặt NequIP từ mã nguồn chính thức
Cài đặt trực tiếp phiên bản ổn định của NequIP từ GitHub:
```powershell
pip install git+https://github.com/mir-group/nequip.git
```

---

## Kiểm tra cài đặt

Trong môi trường ảo đã được kích hoạt, hãy kiểm tra tính sẵn sàng của các thư viện chính:
```powershell
# Kiểm tra PyTorch và CUDA
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# Kiểm tra e3nn và NequIP (dành cho Part 3)
python -c "import e3nn; import nequip; print('e3nn version:', e3nn.__version__); print('NequIP import OK')"
```

Kết quả mong đợi (có hỗ trợ GPU):
```text
PyTorch: 2.6.0+cu124
CUDA available: True
e3nn version: 0.6.0
NequIP import OK
```

---

## Hướng dẫn chạy các thành phần

Sau khi kích hoạt môi trường ảo `nequip-env`, bạn có thể thực hiện các tác vụ sau:

### 1. Khởi chạy Backend Server (FastAPI)
Backend phục vụ các API cho web demo của cả 3 phần. Khi khởi chạy, backend sẽ tự động load mô hình PointNet và NequIP lên **CPU** (để đảm bảo tính tương thích tối đa và tránh lỗi driver CUDA trên các máy chạy demo).
```powershell
cd backend
python -m uvicorn main:app --reload --port 8000
```
Lúc này, server FastAPI sẽ hoạt động tại `http://localhost:8000`.

### 2. Chạy huấn luyện PointNet (Part 2)
Bạn có thể chạy script huấn luyện PointNet trên dữ liệu ModelNet40:
```powershell
cd ai_core
python scripts/train_pointnet.py
```

### 3. Chạy huấn luyện NequIP (Part 3)
Đầu tiên, tải và chuẩn bị tập dữ liệu con của QM9 (100 và 1000 mẫu):
```powershell
cd ai_core
python scripts/prepare_qm9_subsets.py
```
Tiền xử lý dữ liệu sang cấu trúc đồ thị tương thích với NequIP:
```powershell
python scripts/preprocess.py
```
Chạy thử nghiệm huấn luyện toàn bộ các cấu hình ablation study (mạng baseline MLP và mạng tương biến NequIP):
```powershell
./scripts/run_all_experiments.sh
```

---

## Các thư viện chính và vai trò

| Thư viện | Phiên bản | Vai trò trong Đồ án |
|---|---|---|
| `torch` | 2.6.0 | Deep learning framework nền tảng cho MNIST, PointNet và NequIP. |
| `nequip` | git@main | Thư viện mạng GNN tương biến E(3) chính để dự đoán năng lượng QM9 (Part 3). |
| `e3nn` | 0.6.0 | Cung cấp các biểu diễn Tensor tương biến, hài cầu (Spherical Harmonics) cho NequIP. |
| `torch-geometric` | 2.7.0 | Framework xây dựng và huấn luyện Graph Neural Networks. |
| `ase` | 3.28.0 | Đọc, ghi và thao tác trên cấu trúc nguyên tử (Atomistic Simulation Environment). |
| `fastapi` & `uvicorn` | * | Xây dựng và khởi chạy API server cho Web Demo ứng dụng. |
| `numpy` | 2.4.6 | Xử lý mảng dữ liệu và tính toán số học. |
| `matplotlib` | 3.10.9 | Vẽ biểu đồ huấn luyện và so sánh kết quả thực nghiệm. |
| `pytorch-lightning` | 2.6.5 | Đơn giản hóa vòng lặp huấn luyện có cấu trúc. |

---

## Lưu ý quan trọng

1. **Mặc định chạy trên CPU**: Khi khởi chạy backend demo (`backend/main.py`), biến môi trường `CUDA_VISIBLE_DEVICES` được đặt bằng rỗng (`""`) để ép PyTorch chạy trên **CPU**. Điều này giúp tránh xung đột hoặc lỗi crash do driver NVIDIA không tương thích trên máy chạy demo. Đối với việc huấn luyện trong `ai_core`, hãy dùng GPU để tối ưu hóa hiệu năng.
2. **Không commit `.venv`**: Thư mục chứa môi trường ảo rất nặng. Hãy đảm bảo chỉ cập nhật và commit file `requirements.txt` khi cài thêm package mới.
3. **Tắt môi trường ảo**: Để tắt môi trường ảo khi kết thúc làm việc, gõ lệnh `deactivate` trong terminal.
