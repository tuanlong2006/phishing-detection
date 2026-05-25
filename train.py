"""
train.py
--------
Script huấn luyện chính cho Hybrid Bi-LSTM + XGBoost Phishing Detector.

Cách chạy:
    python train.py

Yêu cầu:
    - Dataset đặt tại: data/raw/URL_Classification.csv  (ISCX format)
    - Hoặc chỉnh DATA_PATH và DATA_FORMAT bên dưới

Tham khảo pipeline từ:
    [1] Shahrivari et al. (2020). arXiv:2009.11116
    [2] Le et al. (2018). arXiv:1802.03162
"""

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Tắt log verbose của TensorFlow

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ─── Import các module trong project ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.url_processor import URLProcessor
from src.feature_extractor import extract_batch, FEATURE_NAMES
from src.bilstm_branch import build_bilstm_branch
from src.xgboost_branch import XGBoostBranch
from src.hybrid_model import build_hybrid_model, get_callbacks
from src.utils import (load_iscx_dataset, load_custom_dataset,
                       evaluate_model, print_results_table,
                       plot_confusion_matrix, plot_training_history,
                       plot_feature_importance)


# ═══════════════════════════════════════════════════════════════════════════════
#  CẤU HÌNH — Chỉnh sửa ở đây
# ═══════════════════════════════════════════════════════════════════════════════

# Đường dẫn dataset
DATA_PATH = "data_src_notebooks/processed/processed_urls.csv"

# Format dataset: 'iscx' hoặc 'custom'
DATA_FORMAT = 'custom'

# Hyperparameters
MAX_URL_LEN  = 200    # Độ dài URL tối đa (ký tự)
EMBED_DIM    = 32     # Embedding dimension
LSTM_UNITS   = 64     # Số đơn vị LSTM mỗi chiều
DROPOUT      = 0.3    # Dropout rate
EPOCHS       = 20     # Số epoch tối đa (EarlyStopping sẽ dừng sớm hơn)
BATCH_SIZE   = 64     # Batch size
TEST_SIZE    = 0.2    # Tỷ lệ test set
VAL_SIZE     = 0.1    # Tỷ lệ validation set (từ train set)
RANDOM_STATE = 42

# Thư mục output
OUTPUT_DIR   = "outputs"

# ═══════════════════════════════════════════════════════════════════════════════


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    print("=" * 65)
    print("  HYBRID BI-LSTM + XGBOOST — PHISHING URL DETECTOR")
    print("  Ref: Shahrivari et al. (2020) + Le et al. (2018)")
    print("=" * 65)

    # ── 1. Load Dataset ────────────────────────────────────────────────────
    print("\n[1/6] Loading dataset...")
    if not os.path.exists(DATA_PATH):
        print(f"\n⚠️  Không tìm thấy dataset tại: {DATA_PATH}")
        print("Vui lòng:")
        print("  1. Tải ISCX-URL-2016 tại: https://www.unb.ca/cic/datasets/url-2016.html")
        print("  2. Đặt file vào: data/raw/URL_Classification.csv")
        print("\nHoặc dùng script tạo dataset demo:")
        print("  python create_demo_dataset.py")
        sys.exit(1)

    if DATA_FORMAT == 'iscx':
        df = load_iscx_dataset(DATA_PATH)
    else:
        df = load_custom_dataset(DATA_PATH)

    urls = df['url'].tolist()
    labels = df['label'].values

    # ── 2. Chia Train/Val/Test ─────────────────────────────────────────────
    print("\n[2/6] Chia dữ liệu Train/Val/Test...")
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        urls, labels,
        test_size=TEST_SIZE,
        stratify=labels,
        random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=VAL_SIZE / (1 - TEST_SIZE),
        stratify=y_trainval,
        random_state=RANDOM_STATE
    )
    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # ── 3. Xử lý URL (char-level tokenize) ────────────────────────────────
    print("\n[3/6] Tokenize URL theo ký tự...")
    processor = URLProcessor(max_len=MAX_URL_LEN)
    X_seq_train = processor.fit_transform(X_train)
    X_seq_val   = processor.transform(X_val)
    X_seq_test  = processor.transform(X_test)
    vocab_size  = processor.get_vocab_size()

    # ── 4. Trích xuất đặc trưng tĩnh ──────────────────────────────────────
    print("\n[4/6] Trích xuất 15 đặc trưng tĩnh...")
    X_feat_train = extract_batch(X_train)
    X_feat_val   = extract_batch(X_val)
    X_feat_test  = extract_batch(X_test)
    print(f"  Feature matrix shape: {X_feat_train.shape}")

    # ── 5. Train XGBoost Branch ────────────────────────────────────────────
    print("\n[5/6] Huấn luyện XGBoost branch...")
    xgb = XGBoostBranch()
    t0 = time.time()
    xgb.train(X_feat_train, y_train, X_feat_val, y_val)
    xgb_train_time = time.time() - t0

    # Lấy xác suất từ XGBoost để feed vào hybrid
    xgb_prob_train = xgb.get_proba(X_feat_train)
    xgb_prob_val   = xgb.get_proba(X_feat_val)
    xgb_prob_test  = xgb.get_proba(X_feat_test)

    # Đánh giá XGBoost độc lập (để báo cáo)
    xgb_results = xgb.evaluate(X_feat_test, y_test)
    print(f"  XGBoost alone → Acc: {xgb_results['accuracy']:.4f} | F1: {xgb_results['f1']:.4f}")

    # ── 6. Build và Train Hybrid Model ────────────────────────────────────
    print("\n[6/6] Xây dựng và huấn luyện Hybrid Keras model...")
    bilstm_in, bilstm_out = build_bilstm_branch(
        input_len=MAX_URL_LEN,
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        lstm_units=LSTM_UNITS,
        dropout_rate=DROPOUT
    )
    hybrid_model = build_hybrid_model(
        bilstm_in, bilstm_out,
        xgb_feature_dim=1,
        dropout_rate=DROPOUT
    )
    hybrid_model.summary()

    t0 = time.time()
    history = hybrid_model.fit(
        [X_seq_train, xgb_prob_train], y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=([X_seq_val, xgb_prob_val], y_val),
        callbacks=get_callbacks(patience=5),
        verbose=1
    )
    hybrid_train_time = time.time() - t0

    # ── Evaluate ───────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("ĐÁNH GIÁ KẾT QUẢ")
    print("=" * 65)

    t0 = time.time()
    y_prob = hybrid_model.predict([X_seq_test, xgb_prob_test], verbose=0).flatten()
    hybrid_test_time = time.time() - t0
    y_pred = (y_prob >= 0.5).astype(int)

    all_results = [
        evaluate_model(
            "XGBoost (standalone)",
            y_test,
            (xgb_prob_test.flatten() >= 0.5).astype(int),
            train_time=xgb_train_time
        ),
        evaluate_model(
            "Hybrid Bi-LSTM+XGBoost",
            y_test, y_pred,
            train_time=hybrid_train_time,
            test_time=hybrid_test_time
        ),
    ]

    print_results_table(all_results)

    # ── Lưu kết quả ───────────────────────────────────────────────────────
    print("\nLưu model và biểu đồ...")
    os.makedirs(f"{OUTPUT_DIR}/models", exist_ok=True)
    hybrid_model.save(f"{OUTPUT_DIR}/models/hybrid_keras.h5")
    xgb.save(f"{OUTPUT_DIR}/models/xgb_branch.pkl")

    import joblib
    joblib.dump(processor, f"{OUTPUT_DIR}/models/url_processor.pkl")

    plot_confusion_matrix(y_test, y_pred, "Hybrid Model — Confusion Matrix",
                          save_path=f"{OUTPUT_DIR}/confusion_matrix.png")
    plot_training_history(history,
                          save_path=f"{OUTPUT_DIR}/training_history.png")
    plot_feature_importance(xgb.feature_importance(FEATURE_NAMES),
                            save_path=f"{OUTPUT_DIR}/feature_importance.png")

    # Lưu bảng kết quả ra CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"{OUTPUT_DIR}/results.csv", index=False)

    print(f"\n✅ Hoàn tất! Kết quả lưu tại: {OUTPUT_DIR}/")
    print(f"   - confusion_matrix.png")
    print(f"   - training_history.png")
    print(f"   - feature_importance.png")
    print(f"   - results.csv")


if __name__ == "__main__":
    main()
