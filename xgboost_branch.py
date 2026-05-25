"""
xgboost_branch.py
-----------------
Nhánh Machine Learning: XGBoost thật sự xử lý 15 đặc trưng tĩnh.

Đây là sửa lỗi quan trọng so với code gốc của nhóm:
    - Code cũ: dùng MLP (Dense layers) nhưng đặt tên là "xgboost_branch"
    - Code này: dùng XGBoost thật, lấy xác suất output để ghép vào hybrid

Tại sao dùng XGBoost thay vì MLP cho nhánh này?
    Theo [1] (Shahrivari et al. 2020), XGBoost đạt accuracy ~98.3% trên
    tabular features. MLP chỉ đạt ~97%. Với 15 numeric features, tree-based
    methods vẫn vượt trội hơn dense networks.

Tham khảo:
    [1] Shahrivari et al. (2020). arXiv:2009.11116
    Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System.
    KDD 2016. https://dl.acm.org/doi/10.1145/2939672.2939785
"""

import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score


class XGBoostBranch:
    """
    XGBoost classifier để xử lý các đặc trưng tĩnh của URL.

    Cách hoạt động trong hybrid model:
        1. Train XGBoost trên 15 features
        2. Lấy predict_proba() → 1 giá trị xác suất [0, 1]
        3. Giá trị này được ghép vào output của Bi-LSTM
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42
    ):
        """
        Khởi tạo XGBoost với hyperparameter đã được chỉnh sửa.

        Các tham số dựa trên kết quả trong [1] và thực nghiệm với
        ISCX-URL dataset.
        """
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            eval_metric='logloss',
            use_label_encoder=False,
            n_jobs=-1  # Dùng tất cả CPU cores
        )
        self.is_trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None):
        """
        Huấn luyện XGBoost.

        Args:
            X_train: shape (N, 15) — ma trận features
            y_train: shape (N,) — nhãn 0/1
            X_val  : Validation set (tùy chọn, dùng early stopping)
            y_val  : Nhãn validation
        """
        if X_val is not None and y_val is not None:
            self.model.set_params(early_stopping_rounds=20)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train, verbose=False)

        self.is_trained = True
        print(f"[XGBoost] Đã train xong. "
              f"Best iteration: {self.model.best_iteration if hasattr(self.model, 'best_iteration') else 'N/A'}")

    def get_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Trả về xác suất phishing để ghép vào hybrid model.

        Returns:
            np.ndarray shape (N, 1) — xác suất [0, 1] cho class 1 (phishing)
        """
        if not self.is_trained:
            raise RuntimeError("Gọi train() trước khi get_proba().")
        proba = self.model.predict_proba(X)[:, 1]
        return proba.reshape(-1, 1)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Đánh giá model XGBoost độc lập (để debug và báo cáo)."""
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        y_pred = self.model.predict(X_test)
        return {
            'accuracy' : accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall'   : recall_score(y_test, y_pred),
            'f1'       : f1_score(y_test, y_pred),
        }

    def save(self, path: str):
        """Lưu model ra file."""
        joblib.dump(self.model, path)
        print(f"[XGBoost] Đã lưu model: {path}")

    def load(self, path: str):
        """Load model từ file."""
        self.model = joblib.load(path)
        self.is_trained = True
        print(f"[XGBoost] Đã load model: {path}")

    def feature_importance(self, feature_names: list = None) -> dict:
        """Trả về feature importance để phân tích."""
        imp = self.model.feature_importances_
        if feature_names:
            return dict(sorted(zip(feature_names, imp),
                               key=lambda x: x[1], reverse=True))
        return dict(enumerate(imp))
