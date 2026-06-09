# Đồ án: Nhận diện Chữ số MNIST với Tính Bất Biến Xoay (Rotation Invariance)

Đây là bản Demo ứng dụng (Phần 1) của bài báo cáo Học máy Hình học (Geometric Machine Learning). Mục tiêu là chứng minh sức mạnh của phương pháp **Frame Averaging** so với mạng **CNN Cơ sở (Baseline)** và phương pháp **Tăng cường Dữ liệu (Data Augmentation)**.

## Cấu trúc thư mục
- `/frontend`: Web giao diện (Next.js + Tailwind CSS) theo phong cách Neo-Brutalism.
- `/backend`: API Server (FastAPI + PyTorch) thực hiện suy luận mô hình và trả kết quả.
- `/notebooks`: File Jupyter Notebook huấn luyện các mô hình AI (`train_models.ipynb`).

## Hướng dẫn Khởi chạy
### 1. Huấn luyện Mô hình AI
Để giao diện web hoạt động với AI thật, bạn cần huấn luyện 2 mô hình CNN trước.
1. Mở thư mục `notebooks` và mở file `train_models.ipynb`.
2. Chạy tất cả các ô lệnh (Run All) trong file này. 
3. Code sẽ tự tải MNIST và huấn luyện 2 mô hình, sau đó xuất ra thư mục `backend/models/`.

### 2. Khởi chạy Backend (FastAPI)
Mở một Terminal mới, di chuyển vào thư mục `backend`:
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload
```
Server sẽ chạy ở địa chỉ: `http://localhost:8000`.

### 3. Khởi chạy Frontend (Next.js)
Mở một Terminal thứ hai, di chuyển vào thư mục `frontend`:
```bash
cd frontend
npm install
npm run dev
```
Trang web sẽ chạy ở địa chỉ: `http://localhost:3000`. Hãy mở trình duyệt, tự vẽ chữ số và xoay để xem sự khác biệt giữa 3 mô hình!
