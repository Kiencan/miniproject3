# Simple Optical Character Recognition (OCR)

Chủ đề: Simple Optical Character Recognition (OCR)
Mô tả:
Trong bài tập lần này, bạn sẽ thực hiện xây dựng một hệ thống nhận diện chữ viết tay sử dụng các kỹ thuật xử lý ảnh và học
máy. Đây là một bài toán quan trọng trong lĩnh vực Thị giác máy tính, có thể ứng dụng trong nhiều lĩnh vực như nhập liệu tự
động, nhận diện tài liệu,... Bạn cần phải sử dụng các kỹ thuật xử lý ảnh để phát hiện vị trí, và sử dụng các mô hình học máy để
nhận diện các ký tự viết tay, đồng thời triển khai hệ thống với giao diện người dùng cơ bản.
Lưu ý: Yêu cầu đề bài chỉ là nhận diện từng ký tự viết tay, không cần phải cả câu.
Yêu cầu:
1. Thu thập và Tiền xử lí dữ liệu
Nguồn dữ liệu: Sử dụng một tập dữ liệu chữ viết tay có sẵn như MNIST, EMNIST (đối với ký tự viết tay đơn giản).
Tiền xử lý dữ liệu:
- Sử dụng các kỹ thuật xử lý ảnh đã học để phát hiện vị trí của chữ viết.
- Huấn luyện 1 mô hình đơn giản để phát hiện vị trí của chữ viết (Optional).
- Cân nhắc việc tăng cường dữ liệu để làm đa dạng dữ liệu khi huấn luyện (Optional).
2. Xây dựng mô hình học máy để nhận diện chữ viết tay
- Dựa trên kiến thức đã học của môn Học máy, các bạn cần xây dựng lại mô hình phân loại ký tự từ đầu (from scratch).
- Lựa chọn, hoặc xây dựng thêm 1 mô hình Deep learning với mục đích tương tự. So sánh hiệu năng của 2 mô hình.

Yêu cầu:
3. Xây dựng giao diện người dùng cơ bản
Thiết kế giao diện: Phát triển một giao diện đơn giản (web, application,...) cho phép người dùng tải lên hình ảnh chữ viết
tay hoặc viết trực tiếp chữ viết tay.
Chức năng chính: Sau khi nhận diện, kết quả sẽ hiển thị dưới dạng văn bản hoặc mã ASCII. Giao diện cần có nút "Reset"
và "Tải ảnh" để tăng tính tiện dụng cho người dùng.
Xử lý lỗi và thông báo: Thêm các thông báo lỗi (nếu hình ảnh tải lên không hợp lệ) và xử lý ngoại lệ để đảm bảo giao
diện hoạt động ổn định.

# Kết quả:
Sau khi xây dựng mô hình học máy Softmax Regression (Logistic), với tập dữ liệu là EMNIST và balanced datasets (tập datasets có chứa cả 
số và chữ viết tay), tôi thu nhận được kết quả là 70% => Không tốt cho việc dự đoán mô hình.

Do đó tôi đã xây dựng thêm một mô hình học sâu CNN, và độ chính xác đã tăng lên đáng kể: 90%.

Chi tiết hai mô hình có thể xem trong file `logistic_model.py` hoặc `test.ipynb`.

Sau đây là một vài hình ảnh minh họa khi deploy web app:

![Dự đoán bởi Logistic](https://github.com/user-attachments/assets/4d29dccb-ec61-4c5e-9284-8fcc851fb7f3)

![Dự đoán bởi CNN](https://github.com/user-attachments/assets/552838d0-dd5d-4005-a68a-9b0b4860c30c)

![Dự đoán bởi Logistic](https://github.com/user-attachments/assets/d4f9a4a2-b27f-4e7d-b5bd-b05982f40f67)

![Dự đoán bởi CNN](https://github.com/user-attachments/assets/83367a7d-d1ce-4a13-b8a0-93023525c2a8)

