"""
utils.py
--------
Các hàm tiện ích: đánh giá, vẽ biểu đồ, load dataset.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Không cần GUI display
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
)


# ─── Dataset Loading ──────────────────────────────────────────────────────────

def load_iscx_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load và chuẩn hóa ISCX-URL-2016 dataset.

    Format gốc của ISCX:
        url, type  (type: benign/phishing/malware/defacement)

    Hàm này:
        - Chỉ giữ benign và phishing
        - Đổi nhãn: benign=0, phishing=1
        - Bỏ duplicate, bỏ NaN

    Args:
        csv_path: Đường dẫn đến file CSV

    Returns:
        DataFrame với 2 cột: 'url' và 'label' (0/1)
    """
    df = pd.read_csv(csv_path)

    # Tìm tên cột đúng (ISCX có thể dùng 'url'/'URL' và 'type'/'label')
    url_col = next((c for c in df.columns if c.lower() == 'url'), None)
    type_col = next((c for c in df.columns if c.lower() in ['type', 'label', 'class']), None)

    if url_col is None or type_col is None:
        raise ValueError(f"Không tìm thấy cột url/type. Columns hiện có: {list(df.columns)}")

    df = df[[url_col, type_col]].rename(columns={url_col: 'url', type_col: 'type'})
    df = df.dropna()
    df = df.drop_duplicates(subset='url')

    # Chuyển nhãn text → số
    # Nếu dataset có nhiều loại (malware, defacement,...), gom thành binary
    df['label'] = df['type'].apply(lambda x: 0 if str(x).lower() == 'benign' else 1)
    df = df[['url', 'label']]

    print(f"[Dataset] Loaded: {len(df)} samples")
    print(f"  Legit   (0): {(df.label == 0).sum():,}")
    print(f"  Phishing(1): {(df.label == 1).sum():,}")
    return df



def load_custom_dataset(path):
    import pandas as pd

    df = pd.read_csv(path)

    print("Columns in dataset:", df.columns.tolist())

    # tự tìm cột URL
    if "url" in df.columns:
        url_col = "url"
    elif "URL" in df.columns:
        url_col = "URL"
    elif "processed_url" in df.columns:
        url_col = "processed_url"
    elif "clean_url" in df.columns:
        url_col = "clean_url"

    else:
        raise ValueError("Không tìm thấy cột URL trong dataset.")

    # tự tìm cột label
    if "label" in df.columns:
        label_col = "label"
    elif "type" in df.columns:
        df["label"] = df["type"].apply(lambda x: 0 if x == "benign" else 1)
        label_col = "label"
    elif "Label" in df.columns:
        label_col = "Label"
    else:
        raise ValueError("Không tìm thấy cột label/type trong dataset.")

    df = df[[url_col, label_col]].rename(
        columns={url_col: "url", label_col: "label"}
    )

    df = df.dropna()
    df["url"] = df["url"].astype(str)

    return df


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_model(model_name: str,
                   y_true: np.ndarray,
                   y_pred: np.ndarray,
                   train_time: float = 0.0,
                   test_time: float = 0.0) -> dict:
    """
    Tính đầy đủ các metrics.

    Returns:
        dict với accuracy, precision, recall, f1, roc_auc, train_time, test_time
    """
    results = {
        'classifier' : model_name,
        'train_time' : round(train_time, 3),
        'test_time'  : round(test_time, 4),
        'accuracy'   : round(accuracy_score(y_true, y_pred), 6),
        'recall'     : round(recall_score(y_true, y_pred), 6),
        'precision'  : round(precision_score(y_true, y_pred), 6),
        'f1'         : round(f1_score(y_true, y_pred), 6),
    }
    return results


def print_results_table(results_list: list):
    """
    In bảng kết quả theo format giống Table III trong bài báo [1].

    Args:
        results_list: List[dict] từ evaluate_model()
    """
    header = f"{'Classifier':<28} | {'Train(s)':<10} | {'Test(s)':<10} | {'Accuracy':<10} | {'Recall':<10} | {'Precision':<10} | {'F1':<10}"
    sep = "-" * len(header)
    print("\n" + "=" * len(header))
    print("CLASSIFICATION RESULTS")
    print("=" * len(header))
    print(header)
    print(sep)
    for r in results_list:
        print(f"{r['classifier']:<28} | {r['train_time']:<10.3f} | "
              f"{r['test_time']:<10.4f} | {r['accuracy']:<10.6f} | "
              f"{r['recall']:<10.6f} | {r['precision']:<10.6f} | {r['f1']:<10.6f}")
    print("=" * len(header))


# ─── Visualization ────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          title: str = "Confusion Matrix",
                          save_path: str = None):
    """Vẽ confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=['Legit', 'Phishing'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Saved: {save_path}")
    plt.close()


def plot_training_history(history, save_path: str = None):
    """Vẽ đồ thị loss và accuracy qua các epoch."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.set_title('Loss over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(history.history['accuracy'], label='Train Acc')
    if 'val_accuracy' in history.history:
        ax2.plot(history.history['val_accuracy'], label='Val Acc')
    ax2.set_title('Accuracy over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Saved: {save_path}")
    plt.close()


def plot_feature_importance(importance_dict: dict, save_path: str = None):
    """Vẽ biểu đồ feature importance của XGBoost."""
    names = list(importance_dict.keys())
    values = list(importance_dict.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(names, values, color='steelblue', alpha=0.8)
    ax.set_xlabel('Importance Score')
    ax.set_title('XGBoost Branch — Feature Importance')
    ax.invert_yaxis()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Saved: {save_path}")
    plt.close()
