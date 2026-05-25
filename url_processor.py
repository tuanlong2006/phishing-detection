"""
url_processor.py
----------------
Xử lý chuỗi URL thô thành dạng số để đưa vào Bi-LSTM.

Tham khảo:
    Le et al. (2018). URLNet: Learning a URL Representation with Deep Learning
    for Malicious URL Detection. arXiv:1802.03162
    → Ý tưởng character-level tokenization cho URL.
"""

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


class URLProcessor:
    """
    Tokenizer cấp ký tự (character-level) cho chuỗi URL.

    Ví dụ:
        "http://abc.com" → [8, 4, 4, 12, 11, 11, 5, 1, 2, 3, ...]

    Lý do dùng char-level thay vì word-level:
        - URL không tuân theo cú pháp ngôn ngữ tự nhiên
        - Phishing URL thường thay ký tự ('paypa1.com', 'g00gle.com')
        - Char-level bắt được pattern như '-', '@', '//' tốt hơn
    """

    def __init__(self, max_len: int = 200):
        """
        Args:
            max_len: Độ dài tối đa của URL sau padding.
                     200 ký tự đủ cho ~99% URL thực tế.
        """
        self.max_len = max_len
        # char_level=True: mỗi ký tự là 1 token
        # lower=True: chuẩn hóa chữ hoa/thường
        self.tokenizer = Tokenizer(char_level=True, lower=True, oov_token='<unk>')
        self.is_fitted = False

    def fit(self, url_list: list) -> 'URLProcessor':
        """Học bộ từ điển ký tự từ danh sách URL."""
        self.tokenizer.fit_on_texts(url_list)
        self.is_fitted = True
        print(f"[URLProcessor] Vocab size: {self.get_vocab_size()} ký tự")
        return self

    def transform(self, url_list: list) -> np.ndarray:
        """
        Chuyển danh sách URL thành ma trận số nguyên.

        Args:
            url_list: List[str] - danh sách URL thô

        Returns:
            np.ndarray shape (N, max_len) - đã padding/truncating
        """
        if not self.is_fitted:
            raise RuntimeError("Gọi fit() trước khi transform().")
        sequences = self.tokenizer.texts_to_sequences(url_list)
        # padding='post': thêm 0 vào cuối
        # truncating='post': cắt từ cuối nếu quá dài
        return pad_sequences(sequences, maxlen=self.max_len,
                             padding='post', truncating='post')

    def fit_transform(self, url_list: list) -> np.ndarray:
        """Gộp fit + transform."""
        return self.fit(url_list).transform(url_list)

    def get_vocab_size(self) -> int:
        """Số lượng ký tự khác nhau + 1 (cho padding token 0)."""
        return len(self.tokenizer.word_index) + 1
