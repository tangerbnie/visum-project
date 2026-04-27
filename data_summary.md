Data Summary – VietNews Dataset

# 1. Giới thiệu dataset

Dataset được sử dụng trong bài toán tóm tắt văn bản tiếng Việt, bao gồm hai trường chính:

article: nội dung bài báo đầy đủ
abstract: bản tóm tắt tương ứng

Các bài viết thuộc nhiều lĩnh vực tin tức khác nhau, với độ dài lớn và nội dung đa dạng, trong khi phần tóm tắt ngắn gọn và tập trung vào thông tin chính.

# 2. Quy mô dữ liệu
Tập train được sử dụng: 4000 mẫu
Tập validation được sử dụng: 400 mẫu

Dataset gốc có quy mô lớn, tuy nhiên trong phạm vi bài toán, một tập con được sử dụng để phục vụ phân tích và thử nghiệm.

# 3. Phân tích độ dài văn bản
## 3.1 Độ dài article và summary

Kết quả thống kê cho thấy:

Article có độ dài trung bình lớn, với một số bài rất dài
Summary có độ dài ngắn hơn đáng kể

Điều này phản ánh đúng bản chất của bài toán tóm tắt văn bản, trong đó mô hình cần nén thông tin từ văn bản nguồn.

## 3.2 Tỷ lệ nén (compression ratio)
Mean: 0.0838 (~8.4%)
Min: 0.0164 (~1.6%)
Max: 0.9231 (~92.3%)

Tỷ lệ nén trung bình thấp cho thấy summary chỉ giữ lại một phần rất nhỏ nội dung từ article, phù hợp với bài toán tóm tắt trừu tượng (abstractive summarization).

Tuy nhiên, giá trị tối đa rất cao (~92%) cho thấy tồn tại một số mẫu bất thường, trong đó summary gần tương đương article. Những trường hợp này không phản ánh đúng bản chất của bài toán và có thể gây nhiễu cho mô hình.

# 4. Kiểm tra chất lượng dữ liệu (Data Validation)
## 4.1 Dữ liệu rỗng
Không phát hiện article hoặc summary rỗng
→ Dataset có chất lượng tốt, không chứa mẫu vô nghĩa
## 4.2 Độ dài bất thường
Article quá ngắn (<20 từ): 0 mẫu
Article quá dài (>800 từ): 251 mẫu (~6.3%)

Các bài viết quá dài có thể vượt quá giới hạn đầu vào của mô hình (thường khoảng 512 token), dẫn đến việc bị cắt bớt thông tin (truncation).

## 4.3 Trùng lặp dữ liệu
Unique articles: 4000
Total articles: 4000

→ Không tồn tại dữ liệu trùng lặp, đảm bảo tính đa dạng của dataset.

## 4.4 Mẫu không tương ứng (misalignment)

Trong quá trình kiểm tra, phát hiện 4/4000 mẫu (~0.1%) có hiện tượng không tương ứng giữa article và summary.

Các mẫu này thường có:

Article dạng liệt kê hoặc nội dung không đầy đủ
Summary chứa thông tin không xuất hiện trong article

Tỷ lệ này rất nhỏ và không ảnh hưởng đáng kể đến chất lượng tổng thể của dataset.

# 5. Tiền xử lý dữ liệu (Preprocessing)
## 5.1 Các vấn đề trong dữ liệu gốc
Xuất hiện ký tự HTML trong một số mẫu
Khoảng trắng không chuẩn
Tồn tại nhiều token dạng nối bằng dấu gạch dưới (ví dụ: “dự_án”, “chỉ_định”)
## 5.2 Phương pháp xử lý

Các bước tiền xử lý được áp dụng:

Loại bỏ HTML tags
Chuẩn hóa Unicode (NFC)
Thay thế dấu gạch dưới (_) bằng khoảng trắng
Chuẩn hóa khoảng trắng và dấu câu
Loại bỏ ký tự dư thừa

Ngoài ra, các bài viết dài được xử lý bằng cách truncation trong quá trình tokenization để phù hợp với giới hạn đầu vào của mô hình.

# 6. Kết luận

Từ các phân tích trên, có thể rút ra các điểm chính:

Dataset có chất lượng cao, không chứa dữ liệu rỗng hoặc trùng lặp
Độ dài article lớn và không đồng đều, yêu cầu xử lý truncation
Tỷ lệ nén thấp (~8%) phù hợp với bài toán tóm tắt
Tồn tại một số mẫu bất thường nhưng với tỷ lệ rất nhỏ (~0.1%)
Dữ liệu cần được làm sạch để loại bỏ nhiễu và chuẩn hóa văn bản

Nhìn chung, dataset phù hợp cho bài toán tóm tắt văn bản tiếng Việt sau khi áp dụng các bước tiền xử lý cần thiết.
