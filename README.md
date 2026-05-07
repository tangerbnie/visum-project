# 🇻🇳 Hệ thống Tóm tắt Báo chí Tiếng Việt (ViSum)

Dự án xây dựng hệ thống tóm tắt tin tức tiếng Việt tự động, ứng dụng mô hình 
**BARTpho** được fine-tune trên tập dữ liệu **VietNews (144K mẫu)** 
bằng kỹ thuật **QLoRA**.

## 🚀 Model

- **Model mới nhất (5 epochs):** [OrdinaryAI/visum-qlora-5epochs](https://huggingface.co/OrdinaryAI/visum-qlora-5epochs)
- **Model gốc:** [vinai/bartpho-syllable](https://huggingface.co/vinai/bartpho-syllable)

### 📊 Kết quả huấn luyện (5 epochs - QLoRA)

| Epoch | Train Loss | Eval Loss |
|-------|-----------|----------|
| 1     | 2.685     | 1.175    |
| 2     | 2.513     | 1.120    |
| 3     | 2.402     | 1.105    |
| 4     | 2.307     | 1.091    |
| 5     | 2.263     | **1.086** |

- **Phương pháp:** QLoRA 4-bit + LoRA (r=16, alpha=32)
- **Trainable params:** 8.650.752 / 404.465.664 (2.14%)
- **GPU:** NVIDIA RTX 3090 24GB
- **Thời gian:** 7 giờ 44 phút

## 👥 Thành viên

- **Ngọc Thanh:** Huấn luyện & Đánh giá Model
- **Việt Hoàng:** Tiền xử lý dữ liệu & Phân tích
- **Bảo Nghị:** Frontend & Deploy

## 🛠️ Công nghệ

- **Model:** vinai/bartpho-syllable
- **Dataset:** harouzie/vietnews (144K mẫu)
- **Frontend:** Streamlit
- **Deploy:** Hugging Face Spaces

## 📂 Cấu trúc repo
visum-project/
├── training/
│ └── ViSum_Training.py # Script huấn luyện QLoRA 5 epochs
├── Preprocessing_&_Model_Analysis.ipynb # Phân tích dữ liệu & model
├── data_summary.md # Thống kê dataset
├── error_analysis.txt # Phân tích lỗi (định tính)
├── rouge_results.json # Kết quả ROUGE chi tiết
└── README.md

## 🔗 Liên kết

- **Dataset:** [harouzie/vietnews](https://huggingface.co/datasets/harouzie/vietnews)
- **Model:** [OrdinaryAI/visum-qlora-5epochs](https://huggingface.co/OrdinaryAI/visum-qlora-5epochs)
- **Demo:** (sẽ cập nhật)
