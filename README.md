# AIO_Machine_Translation
<img src="https://github.com/dauvannam321/AIO_Machine_Translation/raw/main/images/demo.png" alt="Demo" width="800"/>

## Giới Thiệu

Dự án Dịch Máy sử dụng mô hình Transformer, một kiến trúc mạng nơ-ron hiệu quả cho nhiệm vụ dịch máy. Mô hình được huấn luyện trên tập dữ liệu [IWSLT15-en-vi](https://huggingface.co/datasets/mt_eng_vietnamese), có sẵn từ Hugging Face Datasets. Tập dữ liệu này chứa các cặp câu văn bản tương ứng giữa tiếng Anh và tiếng Việt. Mục tiêu của dự án là xây dựng một hệ thống dịch máy chất lượng cao có khả năng dịch từ tiếng Anh sang tiếng Việt và ngược lại.

<img src="https://github.com/dauvannam321/AIO_Machine_Translation/blob/main/images/transformer_architecture.jpg" alt="Demo" width="800"/>


## Cách Sử Dụng

### Yêu Cầu Hệ Thống

- Python 3.x
- Các thư viện cần thiết khác (liệt kê trong file `requirements.txt`)

### Cài Đặt

1. Clone repository về máy của bạn:

   ```bash
   https://github.com/dauvannam321/AIO_Machine_Translation.git

2. Tải pretrained các model sau và giải nén chúng trong cùng thư mục CS221.O11:
   - [vi_en](https://drive.google.com/drive/folders/1eevQU8FX1a7Zdu1bOgqryP7gELpqdTu3?usp=sharing)
   - [en_vi](https://drive.google.com/drive/folders/14gIGjSUi8FfpGsB0ih5lUR1bDSopDYoP?usp=sharing)
   
3. Cài đặt các dependencies:

   ```bash
   pip install -r requirements.txt

4. Chạy ứng dụng:

   ```bash
   py app.py
