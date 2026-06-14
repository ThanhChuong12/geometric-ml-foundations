# **Project Report: Báo cáo đồ án 3 - Các bước tiến mới trong Geometric Machine Learning**

Tài liệu này cung cấp thông tin chi tiết về phần report của đồ án 3 môn **Nhập môn học máy**, bao gồm cấu trúc thư mục, nội dung các chương và hướng dẫn biên dịch mã nguồn LaTeX thành file PDF.

## **Table of Contents**
- [**1. Directory Structure**](#1-directory-structure)
- [**2. Report Contents**](#2-report-contents)
- [**3. How to Compile**](#3-how-to-compile)
   - [**Cách 1: Sử dụng Overleaf**](#cách-1-sử-dụng-overleaf)
   - [**Cách 2: Biên dịch cục bộ**](#cách-2-biên-dịch-cục-bộ)
   - [**Cách 3: Sử dụng OpenAI Prism**](#cách-3-sử-dụng-openai-prism)

## **1. Directory Structure**

Thư mục `report/` được tổ chức thành các phần riêng biệt để quản lý mã nguồn LaTeX và file kết quả:

```text
report/
├── src/                 # Thư mục chứa toàn bộ mã nguồn LaTeX
│   ├── chapters/        # Các file .tex chứa nội dung chi tiết từng chương (từ Chương 1 đến Chương 7)
│   ├── graphics/        # Hình ảnh, biểu đồ được chèn vào báo cáo
│   ├── refs/            # Thư mục chứa file tài liệu tham khảo (.bib)
│   ├── codespace.sty    # Package định dạng code tùy chỉnh
│   ├── hcmus-report.cls # File template và style định dạng chuẩn (hcmus-report)
│   └── main.tex         # File gốc để biên dịch toàn bộ báo cáo
├── output/              # Thư mục chứa file PDF sau khi biên dịch
│   └── report.pdf       # Kết quả báo cáo cuối cùng
└── README.md            # Tài liệu bạn đang đọc
```
---

## **2. Report Contents**

Báo cáo trình bày chi tiết về quá trình thực hiện đồ án "Các bước tiến mới trong Geometric Machine Learning" cho môn học **Nhập môn học máy** (Báo cáo Đồ án 3). Báo cáo được chia thành các chương chính (nằm trong thư mục `src/chapters/`):

* **Chương 1: Giới thiệu (`01_gioi_thieu.tex`)**: Giới thiệu bối cảnh, động lực thực hiện đồ án, kỷ nguyên của dữ liệu hình học, sự dịch chuyển sang không gian trọng số cũng như mục tiêu và phạm vi của đồ án.
* **Chương 2: Kiến thức nền tảng (`02_kien_thuc_chuan_bi.tex`)**: Trình bày các khái niệm về đối xứng, bất biến, tương biến và cơ sở lý thuyết nhóm (nhóm hoán vị, nhóm tuần hoàn, nhóm quay $SO(d)$, nhóm Euclid $SE(d)$,... và tác động nhóm).
* **Chương 3: Các phương pháp xây dựng mô hình bảo toàn tính đối xứng (`03_mo_hinh_equivariant.tex`)**: Đi sâu vào hai hướng tiếp cận: can thiệp dữ liệu trên các mô hình có sẵn (Data Augmentation, Group Averaging, Frame Averaging) và xây dựng kiến trúc tương biến từ đầu. Phân tích chi tiết hai mô hình NequIP, PointNet, ứng dụng trong Robotics, các giới hạn (phá vỡ tính đối xứng) và so sánh hai hướng tiếp cận.
* **Chương 4: Lý thuyết nâng cao và các hướng nghiên cứu mới (`04_data_aug_va_huong_moi.tex`)**: Phân tích lợi ích về độ phức tạp mẫu, giải quyết chi phí tính toán bằng Expanders, kiểm định tính đối xứng trong dữ liệu, tính đối xứng tham số mạng nơ-ron, tổng quan hàm mất mát và gộp mô hình, học trên không gian trọng số, đối xứng linh hoạt/khám phá đối xứng, mô hình không phụ thuộc số chiều, và hình học vượt đối xứng.
* **Chương 5: Thực nghiệm và kết quả (`05_thuc_nghiem.tex`)**: Trình bày tổng quan thực nghiệm, khảo sát hiệu quả dữ liệu & định luật tỷ lệ, thực nghiệm xấp xỉ lấy trung bình nhóm, học không gian trọng số qua mạng meta, cảnh quan hàm mất mát & kỹ thuật gộp mô hình, kiểm định tính đối xứng dữ liệu và quan sát thực nghiệm trên tập dữ liệu QM9.
* **Chương 6: Đánh giá và kết luận (`06_danh_gia_ket_luan.tex`)**: Thảo luận về việc có nên luôn dùng mô hình equivariant hay không, tổng hợp ưu điểm/hạn chế, đề xuất các hướng nghiên cứu tương lai và kết luận chung, kèm theo phụ lục phân công công việc nhóm.
* **Chương 7: Demo minh họa (`07_demo.tex`)**: Cung cấp chi tiết 3 phần demo thực nghiệm chính:
  * **Part 1**: Nhận diện chữ số viết tay bất biến với phép xoay (MNIST-rot).
  * **Part 2**: Phân loại đám mây điểm ba chiều với kiến trúc PointNet (ModelNet).
  * **Part 3**: Dự đoán năng lượng phân tử với mạng NequIP (QM9).
  Đồng thời thảo luận chung và liên kết kết quả thực nghiệm với lý thuyết đã trình bày.

## **3. How to Compile**

Mã nguồn của báo cáo sử dụng LaTeX với định dạng class `hcmus-report`. Để biên dịch ra file PDF, ta có thể sử dụng các công cụ phổ biến sau:

### **Cách 1: Sử dụng Overleaf**
1. Nén toàn bộ các file và thư mục bên trong `src/` thành một file `.zip`.
2. Tạo một project mới trên hệ thống [Overleaf](https://www.overleaf.com/) và chọn mục **Upload Project**.
3. Upload file `.zip` vừa tạo lên.
4. Mở file `main.tex` làm file chính và nhấn nút **Compile** (hoặc `Ctrl + S`).

### **Cách 2: Biên dịch cục bộ**
Yêu cầu máy tính đã được cài đặt môi trường LaTeX như TeX Live (trên Linux/Mac) hoặc MiKTeX (trên Windows).
1. Mở terminal hoặc command prompt và di chuyển vào thư mục `src/`:
   ```bash
   cd report/src
   ```
2. Chạy lần lượt các lệnh sau (hoặc dùng `latexmk`) để biên dịch `main.tex` và cập nhật mục lục, tài liệu tham khảo:
   ```bash
   pdflatex main.tex
   bibtex main
   pdflatex main.tex
   pdflatex main.tex
   ```
3. File `main.pdf` sẽ được biên dịch thành công trong cùng thư mục `src/`. Ta có thể đổi tên và di chuyển file này sang thư mục `output/` dưới tên `report.pdf` để lưu trữ và nộp bài.

### **Cách 3: Sử dụng OpenAI Prism**
OpenAI Prism là không gian làm việc (workspace) thông minh được tích hợp sẵn các mô hình AI mạnh mẽ, thiết kế chuyên biệt để viết và cộng tác trong nghiên cứu khoa học hoặc tài liệu kỹ thuật bằng ngôn ngữ LaTeX. Công cụ này thường đi kèm miễn phí cho người dùng có tài khoản ChatGPT cá nhân, Business, Team, Enterprise hoặc Education. Bạn có thể tận dụng AI để soát lỗi định dạng LaTeX, mở rộng ý tưởng và quản lý tài liệu ngay trên một giao diện thống nhất.

**Các bước thực hiện chi tiết:**
1. Đảm bảo thư mục của bạn có chứa file mã nguồn chính, các file định nghĩa, và giữ nguyên cấu trúc thư mục. Nén toàn bộ thư mục `src/` dưới định dạng `.zip`.
2. Truy cập vào tính năng Prism của OpenAI và đăng nhập bằng tài khoản ChatGPT thông thường của bạn.
3. Nhấn vào nút **+ New** (Tạo mới) ở góc trên cùng bên phải.
4. Chọn tùy chọn **Import** (hoặc kéo thả file `.zip` trực tiếp vào giao diện All Projects).
5. Khi dự án được tải lên thành công, bạn sẽ thấy cấu trúc file nằm ở bảng điều khiển bên trái. Hãy nhấp vào file `main.tex` để mở nội dung.
6. Bảng ở giữa là trình soạn thảo, tại đây bạn có thể sử dụng AI để chỉnh sửa văn bản, chèn hình, công thức và trích dẫn.
7. Nhấp vào nút **Compile** (Biên dịch) ở trên cùng. Bảng điều khiển bên phải sẽ hiển thị bản xem trước của tài liệu PDF.
