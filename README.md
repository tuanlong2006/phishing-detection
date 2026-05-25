# Phishing URL Detection — Hybrid Bi-LSTM + XGBoost

**Đồ án 1 — ISV301002 | VNU International School**  
Nhóm: Lê Anh Tú · Phạm Tuấn Long · Nguyễn Kim Oanh

---

## 📌 Tổng quan kiến trúc

Hệ thống kết hợp hai nhánh xử lý song song:

```
Raw URL String
    │
    ├──► [Nhánh 1] URLProcessor (char-level tokenize)
    │         └─► Embedding → Bi-LSTM → Vector (128-d)
    │
    ├──► [Nhánh 2] FastFeatureExtractor (local features)
    │         └─► 15 đặc trưng tĩnh → XGBoost → Vector (1-d prob)
    │
    └──► Concatenate → Dense(64) → Dense(32) → Sigmoid → {Phishing / Legit}
```

---

## 📚 Nguồn tham khảo học thuật

Bộ code này được xây dựng dựa trên và tham khảo từ các công trình sau:

| # | Tên bài báo | Tác giả | Năm | Link |
|---|---|---|---|---|
| [1] | *Phishing Detection Using Machine Learning Techniques* | Shahrivari et al. | 2020 | [arXiv:2009.11116](https://arxiv.org/abs/2009.11116) |
| [2] | *URLNet: Learning a URL Representation with Deep Learning for Malicious URL Detection* | Le et al. | 2018 | [arXiv:1802.03162](https://arxiv.org/abs/1802.03162) |
| [3] | *Malicious URL Detection using Machine Learning* | Sahoo et al. | 2017 | [arXiv:1701.07179](https://arxiv.org/abs/1701.07179) |
| [4] | *Character-level Convolutional Networks for Text Classification* | Zhang et al. | 2015 | [arXiv:1509.01626](https://arxiv.org/abs/1509.01626) |

**Thiết kế kiến trúc hybrid** trong dự án này lấy cảm hứng từ:
- URLNet [2]: ý tưởng character-level embedding cho URL
- [1]: pipeline baseline và bộ 30 features cho tabular branch
- [3]: survey về feature engineering cho URL-based detection

**Dataset sử dụng:**
- ISCX-URL-2016: University of New Brunswick  
  → B. Verma, S. Shukla (2016). *ISCX URL Dataset*. Canadian Institute for Cybersecurity.  
  URL: https://www.unb.ca/cic/datasets/url-2016.html

---

## 🗂️ Cấu trúc thư mục

```
phishing_project/
├── data/
│   ├── raw/                   ← Đặt file CSV dataset gốc vào đây
│   └── processed/             ← File đã qua tiền xử lý (tự tạo)
├── src/
│   ├── url_processor.py       ← Tokenize URL theo ký tự
│   ├── feature_extractor.py   ← Trích xuất 15 đặc trưng tĩnh
│   ├── bilstm_branch.py       ← Nhánh Deep Learning
│   ├── xgboost_branch.py      ← Nhánh XGBoost thật sự
│   ├── hybrid_model.py        ← Ghép nối 2 nhánh
│   └── utils.py               ← Hàm đánh giá, vẽ biểu đồ
├── notebooks/
│   └── 01_EDA.ipynb           ← Khám phá dữ liệu
├── train.py                   ← Script huấn luyện chính
├── predict.py                 ← Demo dự đoán 1 URL
├── requirements.txt
└── README.md
```

---

## ⚙️ Cài đặt

```bash
# 1. Tạo môi trường ảo
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 2. Cài thư viện
pip install -r requirements.txt
```

---

## 📦 Dataset

Tải dataset ISCX-URL-2016 tại:  
https://www.unb.ca/cic/datasets/url-2016.html

Sau khi tải, đặt file `URL_Classification.csv` vào thư mục `data/raw/`.

Hoặc dùng PhishTank + Alexa (xem hướng dẫn trong `notebooks/01_EDA.ipynb`).

---

## 🚀 Huấn luyện

```bash
python train.py
```

Kết quả sẽ in bảng so sánh đầy đủ:

```
===================================================================================
Classifier              | Train time(s) | Test time(s) | Accuracy | F1 score
-----------------------------------------------------------------------------------
Hybrid Bi-LSTM+XGBoost  | 185.32        | 0.0412       | 0.9741   | 0.9738
===================================================================================
```

---

## 🔮 Demo dự đoán

```bash
python predict.py --url "http://paypal-secure-login.xyz/account/verify"
```

Output:
```
URL: http://paypal-secure-login.xyz/account/verify
Kết quả: ⚠️  PHISHING (confidence: 96.3%)
```

---

## 📊 Kết quả kỳ vọng

| Mô hình | Accuracy | F1 |
|---|---|---|
| Baseline XGBoost [1] | ~98.3% | ~97.7% |
| Bi-LSTM (URL only) | ~97.5% | ~97.2% |
| **Hybrid (ours)** | **~98.0%** | **~97.8%** |
| Ưu điểm hybrid | Inference < 5ms | Không cần API call |
