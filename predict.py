"""
predict.py
----------
Demo dự đoán 1 URL bất kỳ sau khi đã train xong.

Cách dùng:
    python predict.py --url "http://paypal-secure-login.xyz/account/verify"
    python predict.py --url "https://google.com"
    python predict.py  # Chạy với một số URL mẫu
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import joblib
from tensorflow.keras.models import load_model


def load_trained_model(output_dir: str = "outputs"):
    """Load model đã train."""
    model_dir = f"{output_dir}/models"

    if not os.path.exists(f"{model_dir}/hybrid_keras.h5"):
        print("❌ Chưa train model! Chạy: python train.py")
        sys.exit(1)

    keras_model = load_model(f"{model_dir}/hybrid_keras.h5")
    xgb_model   = joblib.load(f"{model_dir}/xgb_branch.pkl")
    processor   = joblib.load(f"{model_dir}/url_processor.pkl")

    return keras_model, xgb_model, processor


def predict_url(url: str, keras_model, xgb_model, processor) -> dict:
    """
    Dự đoán 1 URL.

    Returns:
        dict với 'label', 'confidence', 'verdict'
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.feature_extractor import extract_features

    # Xử lý URL
    X_seq = processor.transform([url])
    X_feat = np.array([extract_features(url)], dtype=np.float32)
    xgb_prob = xgb_model.predict_proba(X_feat)[:, 1].reshape(-1, 1)

    # Dự đoán
    prob = float(keras_model.predict([X_seq, xgb_prob], verbose=0).flatten()[0])
    label = 1 if prob >= 0.5 else 0

    return {
        'url'       : url,
        'label'     : label,
        'confidence': prob if label == 1 else 1 - prob,
        'verdict'   : '⚠️  PHISHING' if label == 1 else '✅ LEGITIMATE'
    }


def main():
    parser = argparse.ArgumentParser(description='Phishing URL Detector')
    parser.add_argument('--url', type=str, default=None, help='URL cần kiểm tra')
    parser.add_argument('--output-dir', type=str, default='outputs')
    args = parser.parse_args()

    print("Loading model...")
    keras_model, xgb_model, processor = load_trained_model(args.output_dir)

    if args.url:
        urls_to_test = [args.url]
    else:
        # URL mẫu để demo
        urls_to_test = [
            "https://www.google.com",
            "https://github.com/login",
            "http://paypal-secure-login.xyz/account/verify",
            "http://192.168.1.1/bank/login.php?user=admin",
            "https://bit.ly/3abc123",
            "http://www.amazon.com.security-update.tk/signin",
        ]
        print("\n📋 Chạy demo với các URL mẫu:\n")

    print("\n" + "=" * 70)
    print(f"{'URL':<45} | {'Verdict':<15} | {'Confidence':>10}")
    print("-" * 70)

    for url in urls_to_test:
        result = predict_url(url, keras_model, xgb_model, processor)
        url_display = url[:43] + '..' if len(url) > 45 else url
        print(f"{url_display:<45} | {result['verdict']:<15} | {result['confidence']:>9.1%}")

    print("=" * 70)


if __name__ == "__main__":
    main()
