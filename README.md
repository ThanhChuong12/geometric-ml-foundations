# Geometric Machine Learning Foundations

Dự án nghiên cứu và demo trực quan các nền tảng **Geometric Machine Learning**, tập trung vào tính bất biến (invariance) và đẳng biến (equivariance) trong mạng nơ-ron. Ứng dụng web cho phép tương tác trực tiếp với 2 bài demo:

* **Part 1 — Frame Averaging (2D):** So sánh 3 phương pháp xử lý bất biến xoay trên MNIST: Baseline CNN, Data Augmentation CNN, và Frame Averaging CNN.
* **Part 2 — PointNet (3D):** Phân loại đám mây điểm 3D (ModelNet40) bằng PointNet Basic vs PointNet Full (có T-Net), minh họa Critical Point Set theo Theorem 2.

---

## Tech Stack

| Layer | Công nghệ |
|-------|-----------|
| **Frontend** | Next.js 16, React 19, TypeScript, Tailwind CSS 4, Three.js, Recharts |
| **Backend** | FastAPI, Uvicorn, PyTorch, TorchVision, Pydantic |
| **AI Training** | Jupyter Notebook, PyTorch |

---

## Cấu trúc thư mục

```
geometric-ml-foundations/
│
├── frontend/                   # Web giao diện (Next.js + Tailwind CSS)
│   ├── src/
│   │   ├── app/                # Pages & layouts (App Router)
│   │   ├── components/         # UI components (DigitCanvas, PointCloudViewer, ...)
│   │   ├── services/           # API client (axios)
│   │   └── types/              # TypeScript type definitions
│   └── public/                 # Static assets
│
├── backend/                    # API Server (FastAPI + PyTorch)
│   ├── main.py                 # Entrypoint — FastAPI app
│   ├── api/                    # Route handlers
│   │   ├── part1_routes.py     # POST /api/part1/predict (MNIST digit)
│   │   └── part2_routes.py     # POST /api/part2/classify, GET sample, ...
│   ├── core/                   # Model architectures (thuần PyTorch)
│   │   ├── frame_averaging.py  # SimpleCNN, FrameAveragingCNN
│   │   └── pointnet_model.py   # PointNetBasic, PointNetFull, T-Net
│   ├── models/                 # Pydantic schemas (request/response)
│   │   ├── schemas.py          # Part 1 schemas
│   │   └── pointnet_schemas.py # Part 2 schemas
│   ├── services/               # Business logic & inference
│   │   ├── model_service.py    # Part 1: preprocessing ảnh, inference 3 model CNN
│   │   └── pointnet_service.py # Part 2: preprocessing point cloud, inference PointNet
│   └── requirements.txt
│
├── ai_core/                    # Huấn luyện & đánh giá mô hình AI
│   ├── notebooks/
│   │   ├── train_models.ipynb          # Train 3 CNN cho Part 1 (MNIST)
│   │   └── pointnet_classification.ipynb # Train PointNet cho Part 2 (ModelNet40)
│   ├── scripts/
│   │   ├── train_full_v2.py    # Script train Part 1 (CLI)
│   │   ├── train_pointnet.py   # Script train Part 2 (CLI)
│   │   ├── download_assets.py  # Download sample point clouds & classes
│   │   └── preprocess.py       # Tiền xử lý dữ liệu phân tử (NequIP)
│   ├── train.py                # Entry point huấn luyện
│   └── evaluate.py             # Entry point đánh giá
│
├── storage/                    # File tĩnh (dữ liệu, trọng số) — KHÔNG nằm trong backend
│   ├── weights/                # Pretrained model weights (.pth)
│   │   ├── model_baseline.pth
│   │   ├── model_augmented.pth
│   │   ├── model_frame.pth
│   │   ├── pointnet_cls.pth
│   │   └── pointnet_basic.pth
│   ├── data/                   # Dữ liệu demo
│   │   ├── modelnet40_classes.json
│   │   └── sample_clouds/      # File .npy point cloud mẫu
│   └── debug_images/           # Ảnh debug preprocessing (auto-generated)
│
├── docs/                       # Tài liệu kỹ thuật, paper tham khảo, slide
├── report/                     # Báo cáo LaTeX
├── nequip-env/                 # Môi trường NequIP (thí nghiệm riêng)
└── README.md
```

---

## Hướng dẫn Khởi chạy

### 1. Huấn luyện Mô hình AI

Để giao diện web hoạt động với AI thật, bạn cần huấn luyện mô hình trước. Weights sau khi train sẽ được lưu vào `storage/weights/`.

#### Part 1 — Frame Averaging (MNIST)
Mở `ai_core/notebooks/train_models.ipynb` và **Run All**. Code sẽ tự tải MNIST và huấn luyện 3 mô hình:
- `model_baseline.pth` — CNN thuần (không xử lý xoay)
- `model_augmented.pth` — CNN + Data Augmentation
- `model_frame.pth` — CNN + Frame Averaging (PCA)

Hoặc chạy bằng CLI:
```bash
cd ai_core
python scripts/train_full_v2.py
```

#### Part 2 — PointNet (ModelNet40)
Mở `ai_core/notebooks/pointnet_classification.ipynb` và **Run All**, hoặc:
```bash
cd ai_core
python scripts/train_pointnet.py
```
Sau đó tải sample point clouds cho demo:
```bash
python scripts/download_assets.py
```

### 2. Khởi chạy Backend (FastAPI)

Mở một Terminal, di chuyển vào thư mục `backend`:
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload
```
Server sẽ chạy ở địa chỉ: **http://localhost:8000**

API Endpoints:
| Method | Endpoint | Mô tả |
|--------|----------|--------|
| `POST` | `/api/part1/predict` | Nhận dạng chữ số viết tay (MNIST) |
| `POST` | `/api/part2/classify` | Phân loại point cloud 3D |
| `GET` | `/api/part2/sample/{class}` | Lấy point cloud mẫu theo class |
| `GET` | `/api/part2/classes` | Danh sách 40 class ModelNet40 |
| `GET` | `/api/part2/status` | Health check & trạng thái model |

### 3. Khởi chạy Frontend (Next.js)

Mở một Terminal thứ hai, di chuyển vào thư mục `frontend`:
```bash
cd frontend
npm install
npm run dev
```
Trang web sẽ chạy ở địa chỉ: **http://localhost:3000**

Mở trình duyệt và trải nghiệm:
- 🖊️ **Part 1:** Vẽ chữ số trên canvas, xoay để xem sự khác biệt giữa 3 phương pháp.
- 🔷 **Part 2:** Chọn vật thể 3D, xoay/thêm nhiễu/bỏ điểm để so sánh PointNet Basic vs Full.

---

## Tài liệu tham khảo

* **Frame Averaging Paper:** Puny et al., *"Frame Averaging for Invariant and Equivariant Network Design"*, ICLR 2022
* **PointNet Paper:** Qi et al., *"PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation"*, CVPR 2017
* Slide bài giảng và tài liệu chi tiết: xem thư mục `docs/`
