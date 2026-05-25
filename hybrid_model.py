"""
hybrid_model.py
---------------
Ghép nhánh Bi-LSTM và XGBoost thành một mô hình hybrid hoàn chỉnh.

Kiến trúc:
    ┌─────────────────────┐    ┌─────────────────────┐
    │  URL (char sequence) │    │  URL (15 features)  │
    └─────────┬───────────┘    └──────────┬──────────┘
              │                           │
         Bi-LSTM                      XGBoost
              │ 64-d vector                │ 1-d prob
              └─────────────┬─────────────┘
                       Concatenate (65-d)
                            │
                       Dense(64, relu)
                       BatchNorm
                       Dropout(0.3)
                            │
                       Dense(32, relu)
                            │
                       Dense(1, sigmoid)
                            │
                      {0: legit, 1: phishing}

Nguồn tham khảo thiết kế:
    Ý tưởng hybrid từ: Sahoo et al. (2017). arXiv:1701.07179
    Fusion layer từ: Thực nghiệm trên ISCX dataset
"""

import numpy as np
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def build_hybrid_model(
    bilstm_input: object,
    bilstm_output: object,
    xgb_feature_dim: int = 1,
    dropout_rate: float = 0.3
) -> Model:
    """
    Xây dựng Keras model phần fusion + decision layer.

    Args:
        bilstm_input  : Input tensor từ Bi-LSTM branch
        bilstm_output : Output tensor (64-d) từ Bi-LSTM branch
        xgb_feature_dim: Số chiều từ XGBoost (thường = 1, là xác suất)
        dropout_rate  : Dropout rate cho decision layers

    Returns:
        Keras Model compiled và sẵn sàng train
    """
    # --- Input cho phần XGBoost probability ---
    xgb_input = layers.Input(shape=(xgb_feature_dim,), name='xgb_prob_input')

    # --- Fusion ---
    merged = layers.Concatenate(name='fusion')([bilstm_output, xgb_input])

    # --- Decision Layers ---
    z = layers.Dense(64, activation='relu', name='decision_1')(merged)
    z = layers.BatchNormalization(name='batch_norm')(z)
    z = layers.Dropout(dropout_rate, name='decision_dropout')(z)
    z = layers.Dense(32, activation='relu', name='decision_2')(z)

    # --- Output ---
    output = layers.Dense(1, activation='sigmoid', name='output')(z)

    # --- Compile ---
    model = Model(
        inputs=[bilstm_input, xgb_input],
        outputs=output,
        name='Hybrid_BiLSTM_XGBoost'
    )

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def get_callbacks(patience: int = 5) -> list:
    """
    Callbacks để kiểm soát quá trình training.

    Args:
        patience: Số epoch chờ trước khi dừng sớm

    Returns:
        List callbacks cho model.fit()
    """
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    return [early_stop, reduce_lr]


class HybridPhishingDetector:
    """
    Wrapper class bao gồm toàn bộ pipeline:
    URLProcessor + Bi-LSTM + XGBoostBranch + Fusion Model.

    Dùng class này để train, evaluate, và predict.
    """

    def __init__(self, max_url_len: int = 200, embed_dim: int = 32,
                 lstm_units: int = 64, dropout_rate: float = 0.3):
        from src.url_processor import URLProcessor
        from src.feature_extractor import extract_batch
        from src.bilstm_branch import build_bilstm_branch
        from src.xgboost_branch import XGBoostBranch

        self.max_url_len = max_url_len
        self.embed_dim = embed_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate

        self.url_processor = URLProcessor(max_len=max_url_len)
        self.xgb_branch = XGBoostBranch()
        self.keras_model = None
        self._extract_batch = extract_batch

    def fit(self, urls_train: list, y_train: np.ndarray,
            urls_val: list = None, y_val: np.ndarray = None,
            epochs: int = 20, batch_size: int = 64):
        """
        Huấn luyện toàn bộ pipeline.

        Args:
            urls_train: List URL training
            y_train   : Nhãn 0/1 (0=legit, 1=phishing)
            urls_val  : List URL validation (tùy chọn)
            y_val     : Nhãn validation
            epochs    : Số epoch tối đa
            batch_size: Batch size
        """
        from src.bilstm_branch import build_bilstm_branch

        print("=" * 60)
        print("BƯỚC 1: Xử lý chuỗi URL...")
        X_seq_train = self.url_processor.fit_transform(urls_train)

        print("BƯỚC 2: Trích xuất đặc trưng tĩnh...")
        X_feat_train = self._extract_batch(urls_train)

        print("BƯỚC 3: Huấn luyện XGBoost branch...")
        if urls_val is not None:
            X_feat_val = self._extract_batch(urls_val)
            self.xgb_branch.train(X_feat_train, y_train, X_feat_val, y_val)
        else:
            self.xgb_branch.train(X_feat_train, y_train)

        print("BƯỚC 4: Lấy XGBoost probabilities...")
        xgb_prob_train = self.xgb_branch.get_proba(X_feat_train)

        print("BƯỚC 5: Xây dựng và huấn luyện Keras hybrid model...")
        bilstm_in, bilstm_out = build_bilstm_branch(
            input_len=self.max_url_len,
            vocab_size=self.url_processor.get_vocab_size(),
            embed_dim=self.embed_dim,
            lstm_units=self.lstm_units,
            dropout_rate=self.dropout_rate
        )

        self.keras_model = build_hybrid_model(
            bilstm_in, bilstm_out,
            xgb_feature_dim=1,
            dropout_rate=self.dropout_rate
        )

        self.keras_model.summary()

        # Chuẩn bị validation data nếu có
        val_data = None
        if urls_val is not None and y_val is not None:
            X_seq_val = self.url_processor.transform(urls_val)
            xgb_prob_val = self.xgb_branch.get_proba(self._extract_batch(urls_val))
            val_data = ([X_seq_val, xgb_prob_val], y_val)

        self.keras_model.fit(
            [X_seq_train, xgb_prob_train], y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_data,
            callbacks=get_callbacks(patience=5),
            verbose=1
        )

        print("=" * 60)
        print("Huấn luyện hoàn tất!")

    def predict_proba(self, urls: list) -> np.ndarray:
        """Trả về xác suất phishing cho danh sách URL."""
        X_seq = self.url_processor.transform(urls)
        X_feat = self._extract_batch(urls)
        xgb_prob = self.xgb_branch.get_proba(X_feat)
        return self.keras_model.predict([X_seq, xgb_prob], verbose=0).flatten()

    def predict(self, urls: list, threshold: float = 0.5) -> np.ndarray:
        """Trả về nhãn 0/1."""
        return (self.predict_proba(urls) >= threshold).astype(int)

    def save(self, model_dir: str):
        """Lưu toàn bộ model."""
        import os
        os.makedirs(model_dir, exist_ok=True)
        self.keras_model.save(f"{model_dir}/keras_model.h5")
        self.xgb_branch.save(f"{model_dir}/xgb_model.pkl")
        import joblib
        joblib.dump(self.url_processor, f"{model_dir}/url_processor.pkl")
        print(f"[Hybrid] Đã lưu model vào: {model_dir}/")
