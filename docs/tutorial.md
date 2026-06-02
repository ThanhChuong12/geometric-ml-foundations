# Hướng Dẫn Về Các Phát Triển Gần Đây Trong Học Sâu Hình Học (Geometric Deep Learning)

Tài liệu này tổng hợp nội dung từ bài hướng dẫn về các bước tiến mới trong lĩnh vực Học sâu Hình học, với trọng tâm lớn nhất là **Tính đối xứng (Symmetries)**. Tính đối xứng không chỉ xuất hiện trong dữ liệu và tác vụ mà còn tồn tại ngay bên trong chính các mô hình học máy.

---

## Phần 1: Cơ Bản Về Tính Đối Xứng và Xây Dựng Mô Hình

### 1. Tại Sao Cần Quan Tâm Đến Tính Đối Xứng?
Tính đối xứng ảnh hưởng đến nhiều khía cạnh quan trọng của học máy:
- **Hiệu quả dữ liệu (Sample Complexity):** Cần ít dữ liệu hơn để học.
- **Khả năng khái quát hóa (Generalization) và Tính bền vững (Robustness):** Giúp mô hình hoạt động tốt và có thể dự đoán được đối với dữ liệu ngoài phân phối (out-of-distribution).
- **Cảnh quan tối ưu hóa (Optimization Landscape):** Tính đối xứng bên trong mô hình ảnh hưởng đến quá trình tối ưu hóa, cách so sánh và hợp nhất (merge) các mô hình.

### 2. Các Khái Niệm Cơ Bản
- **Tính Bất Biến (Invariance):** Khi áp dụng phép biến đổi lên đầu vào, đầu ra của mô hình không thay đổi. Ví dụ: Dịch chuyển hình ảnh của một con mèo trong khung hình vẫn cho ra kết quả phân loại là "con mèo".
- **Tính Đồng Biến (Equivariance):** Khi áp dụng phép biến đổi lên đầu vào, đầu ra cũng thay đổi theo một cách tương ứng có thể dự đoán được. Ví dụ: Khi dự đoán node embeddings cho một đồ thị, nếu ta hoán vị thứ tự các nút đầu vào thì các embeddings ở đầu ra cũng phải bị hoán vị tương ứng.
- **Lý Thuyết Nhóm (Group Theory):** Các phép biến đổi (như xoay, tịnh tiến, hoán vị) tạo thành một "Nhóm". Một nhóm bao gồm: phần tử định danh, phần tử nghịch đảo và tính đóng (kết hợp hai phép biến đổi thuộc nhóm sẽ tạo ra một phép biến đổi thuộc nhóm). Quỹ đạo (Orbit) là tập hợp tất cả các trạng thái của một điểm dữ liệu khi bị tác động bởi tất cả các phép biến đổi trong nhóm.

### 3. Đưa Tính Đối Xứng Vào Mô Hình (Phương pháp Hộp đen - Black-box)
Sử dụng các mạng neural tiêu chuẩn và tác động ở khía cạnh dữ liệu:
- **Tăng cường dữ liệu (Data Augmentation):** Áp dụng các phép biến đổi lên dữ liệu huấn luyện. Dễ thực hiện nhưng chỉ mang tính xấp xỉ, không đảm bảo tính đối xứng hoàn toàn khi suy luận.
- **Chuẩn hóa (Canonicalization):** Chuyển dữ liệu về một dạng chuẩn duy nhất (ví dụ: sắp xếp các chuỗi) trước khi đưa vào mạng. Nhược điểm là đôi khi không thể tìm được dạng chuẩn liên tục (gây ra sự cố về tính liên tục khi hai đầu vào giống nhau bị ánh xạ thành hai dạng chuẩn khác xa nhau).
- **Lấy trung bình nhóm (Group Averaging):** Áp dụng tất cả các phép biến đổi của nhóm lên một điểm dữ liệu, đưa qua mô hình rồi tính trung bình kết quả. Rất chính xác và liên tục, nhưng chi phí tính toán cực lớn nếu kích thước nhóm quá lớn.
- **Lấy trung bình khung (Frame Averaging):** Là phương pháp dung hòa, chọn ra một tập con các phép biến đổi phụ thuộc vào đặc điểm của điểm dữ liệu đó (ví dụ: dùng PCA để xác định các phép xoay chính) và lấy trung bình trên tập con này.

### 4. Xây Dựng Kiến Trúc Mô Hình Đồng Biến (Equivariant Architectures)
Xây dựng mô hình từ đầu bằng cách thiết kế các lớp (layers) tôn trọng tính đối xứng:
- **Các Lớp Tuyến Tính Đồng Biến:** Yêu cầu mô hình phải chia sẻ trọng số (weight sharing) dựa trên cấu trúc đối xứng (ví dụ: CNN cho tính đồng biến với phép dịch chuyển, Deep Sets cho hoán vị). Số lượng tham số tự do bị giới hạn nhiều.
- **Tích Chập Nhóm (Group Convolution):** Mở rộng phép tích chập thông thường bằng cách thay thế phép dịch chuyển bằng các phép toán của nhóm mục tiêu.
- **Lý Thuyết Bất Biến và Đa Thức (Invariant Theory):** Xây dựng các đa thức bậc thấp (như khoảng cách tương đối, góc, tích vô hướng) làm đặc trưng đầu vào cho mô hình, đảm bảo tính bất biến/đồng biến và đủ tính phổ quát (universality). Tương tự, Representation Theory sử dụng dạng Tensor bậc cao để có các lớp tuyến tính mạnh mẽ hơn.
- **Ứng dụng:** Mô phỏng hệ thống nguyên tử (dự đoán lực, năng lượng từ cấu trúc đồ thị nguyên tử) và trong Robotics (điều khiển robot với các hành động đồng biến với phép xoay/tịnh tiến).
- **Hạn chế (Symmetry Breaking):** Mô hình đồng biến hoàn hảo đôi khi không thể tạo ra các kết quả thiếu tính đối xứng. Do đó, cần có các phương pháp "phá vỡ tính đối xứng" (ví dụ: thêm nhiễu vào trọng số) khi tạo sinh dữ liệu phức tạp.

---

## Phần 2: Các Hướng Đi Mới và Góc Nhìn Lý Thuyết

### 1. Mô Hình Linh Hoạt (Flexibility) và Khám Phá Tính Đối Xứng
Thay vì thiết kế mô hình cứng nhắc cho một bài toán cụ thể, các nghiên cứu đang hướng tới:
- **Mô hình nền tảng đa đối xứng (Contextual World Models / Any-subgroup Equivariant Networks):** Một mô hình duy nhất mã hóa nhiều loại đối xứng khác nhau. Khi hoạt động, mô hình dựa vào ngữ cảnh hoặc các tín hiệu đầu vào điều khiển (steering arguments) để tự động kích hoạt tính bất biến hoặc đồng biến cần thiết.
- **Khám phá tính đối xứng (Symmetry Discovery):** Học trực tiếp các phép biến đổi hoặc cấu trúc đối xứng từ bộ dữ liệu, thay vì được người dùng định nghĩa trước.

### 2. Tính Đối Xứng Bên Trong Trọng Số Mô Hình (Neural Parameter Symmetries)
Bản thân kiến trúc mạng (MLP, Attention, LoRA) có các đối xứng nội tại. Việc thay đổi tỷ lệ (scaling) ở hàm ReLU hay hoán vị thứ tự các nơ-ron không làm thay đổi hàm toán học mà mạng thực hiện.
- **Cảnh quan tối ưu hóa (Loss Landscape):** Các đối xứng liên tục tạo ra các rãnh kết nối giữa các điểm cực tiểu (minima). Ngược lại, các đối xứng rời rạc (hoán vị) khiến các giải pháp bị phân tán. Để hợp nhất hai mô hình (Model Merging), ta bắt buộc phải căn chỉnh (align) các nơ-ron để chúng nằm chung một "thung lũng" tối ưu trước khi lấy trung bình.
- **Meta-networks (Mô hình xử lý mô hình):** Khi ta coi chính các mạng neural là dữ liệu (để dự đoán hiệu suất, nguồn gốc, tự động nén, hay so sánh mô hình), ta cần thiết kế các meta-networks nhận thức được tính đối xứng (bằng cách dùng Graph Neural Networks hoặc Equivariant Transformers) để xử lý các ma trận trọng số một cách chính xác.

### 3. Đóng Góp Lý Thuyết Mới
- **Ứng dụng lý thuyết Đồ thị giãn nở (Expanders) vào Group Averaging:** Thay vì lấy trung bình trên toàn bộ nhóm khổng lồ (ví dụ: $O(N!)$), chỉ cần chọn ngẫu nhiên một tập con có kích thước logarit (vài chục phần tử) là đủ để giữ nguyên lợi ích thống kê và khả năng xấp xỉ.
- **Kiểm định tính đối xứng (Testing Symmetry):** Cung cấp các công cụ toán học ngẫu nhiên hóa để kiểm tra xem phân phối dữ liệu đầu vào có thực sự mang tính đối xứng hay không trước khi quyết định sử dụng mô hình học sâu hình học.
- **Độ phức tạp mẫu (Sample Complexity):** Được chứng minh qua định lý Weyl mở rộng, tính đối xứng về bản chất làm giảm số chiều của không gian đầu vào (thành chiều của không gian thương - quotient space), do đó giảm thiểu đáng kể lượng dữ liệu cần thiết.
- **Mô hình đa chiều (Any-dimensional models):** Khả năng huấn luyện một tập tham số cố định trên các đối tượng nhỏ (như đồ thị ít nút) và áp dụng hiệu quả lên các đối tượng lớn hơn rất nhiều (nhờ vào tính Ổn định Biểu diễn - Representation Stability).
- **Giới hạn của Giả thuyết Đa tạp (Manifold Hypothesis):** Chỉ riêng việc dữ liệu nằm trên một đa tạp số chiều thấp là chưa đủ để đảm bảo mô hình có thể học được. Nếu đa tạp có độ cong không bị chặn dưới bởi số dương (ví dụ: không gian Hyperbolic), mô hình vẫn gặp khó khăn lớn.
- **Học Sâu Tô-pô (Topological Deep Learning):** Đồ thị thông thường và Message Passing GNNs được chứng minh là bất lực trong việc tính toán các đặc trưng tô-pô (như số thành phần liên thông, lỗ hổng). Cần các kiến trúc mới (như Simplicial/Cellular Complexes) để xử lý dữ liệu phức hợp này.

### Tổng Kết
Trong kỷ nguyên của Mô hình Ngôn ngữ Lớn (LLMs), câu hỏi đặt ra là "liệu tính đối xứng có còn cần thiết ở quy mô dữ liệu khổng lồ?". Câu trả lời là có khoảng cách vẫn tồn tại: đối với các tập dữ liệu nhỏ, dữ liệu khoa học đắt đỏ (như phân tử, vật lý), Geometric Deep Learning vẫn là chìa khóa không thể thay thế. Hơn nữa, việc hiểu và tận dụng tính đối xứng xấp xỉ (approximate symmetries) sẽ là hướng đi thực tiễn để áp dụng các lý thuyết hình học vào những môi trường dữ liệu nhiễu và ít chuẩn hóa trong thế giới thực.
