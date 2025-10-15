
# 🈶 kanji-ml-server  
**AI server for Kanji prediction model (CNN + FastAPI)**  

Dự án này triển khai mô hình **CNN (Convolutional Neural Network)** để nhận dạng và dự đoán **chữ Hán (Kanji)**.  
FastAPI được sử dụng để xây dựng REST API và giao diện tương tác trực tiếp qua endpoint `/canvas`.

## ⚙️ Hướng dẫn cài đặt và chạy demo

### 1️⃣ Clone dự án

```bash
git clone https://github.com/nhutphat1203/cnn-kanji.git
cd cnn-kanji
````

### 2️⃣ Tạo và kích hoạt môi trường ảo (virtual environment)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 3️⃣ Cài đặt các thư viện phụ thuộc

```bash
pip install -r requirements.txt
```

### 4️⃣ Chạy FastAPI server

```bash
fastapi dev .\src\main.py
```

Sau khi chạy thành công, terminal sẽ hiển thị:

```
Server running on http://127.0.0.1:8000
```

---

## 🧠 Kiểm tra mô hình

Mở trình duyệt và truy cập:
👉 **[http://127.0.0.1:8000/canvas](http://127.0.0.1:8000/canvas)**

Tại đây bạn có thể **vẽ trực tiếp chữ Kanji**, nhấn **Predict** để gửi ảnh tới server và xem mô hình CNN dự đoán kết quả.



## 🚀 Hướng phát triển

* Mở rộng mô hình để nhận dạng nhiều chữ Kanji phức tạp hơn.
* Hỗ trợ upload ảnh hàng loạt để dự đoán nhiều chữ cùng lúc.
* Tối ưu tốc độ inference bằng TensorRT hoặc ONNX Runtime.
* Bổ sung dashboard thống kê độ chính xác của mô hình.

## 📦 Dữ liệu huấn luyện: ETL9B Dataset

Dự án sử dụng **ETL9B**, một trong những bộ dữ liệu chữ viết tay tiếng Nhật do **ETL Character Database** (thuộc NTT Laboratories, Nhật Bản) phát hành.

### 🧾 Giới thiệu
ETL9B là một tập dữ liệu chữ viết tay gồm các ký tự **Kanji**, **Hiragana**, và **Katakana** được viết bởi nhiều người Nhật khác nhau.  
Bộ dữ liệu này được thiết kế để phục vụ nghiên cứu **nhận dạng ký tự tiếng Nhật (Japanese OCR)**.

### 🔍 Thông tin chi tiết
- **Số lượng ký tự:** 3036 ký tự Kanji chuẩn JIS Level-2 (trong tổng hơn 6000 ký tự tiếng Nhật).  
- **Số người viết:** 200 người (mỗi người viết cùng một bộ ký tự).  
- **Tổng số mẫu ảnh:** khoảng **607,200 hình ảnh** (3036 ký tự × 200 mẫu).  
- **Kích thước ảnh:** 64×63 pixels (bitmap grayscale, 8-bit).  
- **Định dạng gốc:** Binary (chuẩn ETL format `.ETL9B`).  
- **Định dạng chuyển đổi thường dùng:** PNG hoặc NumPy array (để huấn luyện CNN).  

### 🧠 Cách sử dụng trong dự án
Trong dự án **cnn-kanji**, bộ ETL9B được:
1. Giải mã từ file nhị phân ETL9B gốc sang ảnh grayscale
2. Chuẩn hóa cường độ pixel về khoảng `[0, 1]`.  
3. Chuyển đổi nhãn (label) từ mã JIS sang mã Unicode để mô hình CNN có thể dự đoán ký tự tương ứng.  
4. Chia tập dữ liệu thành:
   - **Training set:** 80%  
   - **Validation set:** 10%  
   - **Test set:** 10%

### ⚠️ Lưu ý bản quyền
ETL9B là dữ liệu công khai cho mục đích **nghiên cứu học thuật và phi thương mại**

Nguồn tham khảo: [ETL Character Database](http://etlcdb.db.aist.go.jp/the-etl-character-database/)
