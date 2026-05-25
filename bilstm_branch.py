"""
bilstm_branch.py
----------------
Nhánh Deep Learning: Bi-LSTM xử lý chuỗi URL ký tự.

Tham khảo kiến trúc từ:
    Le et al. (2018). URLNet: Learning a URL Representation with Deep Learning
    for Malicious URL Detection. arXiv:1802.03162
    → Sử dụng character-level embedding + sequence model cho URL.

    Zhang et al. (2015). Character-level Convolutional Networks for Text
    Classification. arXiv:1509.01626
    → Cơ sở lý thuyết cho character-level approach.

Điều chỉnh so với URLNet gốc:
    - Thay 1D-CNN bằng Bi-LSTM để capture context 2 chiều
    - Thêm Dropout để giảm overfitting (bổ sung so với code gốc của nhóm)
    - Thêm LayerNormalization để ổn định training
"""

from tensorflow.keras import layers, Model


def build_bilstm_branch(
    input_len: int,
    vocab_size: int,
    embed_dim: int = 32,
    lstm_units: int = 64,
    dropout_rate: float = 0.3
) -> tuple:
    """
    Xây dựng nhánh Bi-LSTM.

    Args:
        input_len   : Độ dài chuỗi đầu vào (max_len của URLProcessor)
        vocab_size  : Số lượng ký tự khác nhau (vocab size của tokenizer)
        embed_dim   : Chiều của embedding vector cho mỗi ký tự
        lstm_units  : Số đơn vị trong mỗi chiều LSTM
                      (output sẽ là lstm_units * 2 do Bi-directional)
        dropout_rate: Tỷ lệ dropout để chống overfitting

    Returns:
        (input_tensor, output_tensor): Để dùng với Keras Functional API
    """
    # --- Input ---
    inputs = layers.Input(shape=(input_len,), name='url_char_input')

    # --- Embedding ---
    # Mỗi ký tự được ánh xạ thành vector embed_dim chiều
    # mask_zero=True: bỏ qua padding token (0) khi tính LSTM
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        mask_zero=True,
        name='char_embedding'
    )(inputs)

    # --- Bi-LSTM Layer 1 ---
    # return_sequences=True: trả về output tại mỗi bước thời gian
    # → cần thiết cho việc chồng 2 lớp LSTM
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True, name='lstm_1'),
        name='bilstm_1'
    )(x)
    x = layers.Dropout(dropout_rate, name='dropout_1')(x)

    # --- Bi-LSTM Layer 2 ---
    # return_sequences=False: chỉ lấy output cuối cùng
    x = layers.Bidirectional(
        layers.LSTM(lstm_units // 2, name='lstm_2'),
        name='bilstm_2'
    )(x)
    x = layers.Dropout(dropout_rate, name='dropout_2')(x)

    # --- Projection ---
    # Nén xuống 64-d trước khi concatenate với nhánh XGBoost
    output = layers.Dense(64, activation='relu', name='bilstm_proj')(x)

    return inputs, output


def get_bilstm_output_dim() -> int:
    """Trả về số chiều output của nhánh Bi-LSTM (để tính input cho hybrid)."""
    return 64
