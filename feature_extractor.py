"""
feature_extractor.py
--------------------
Trích xuất các đặc trưng tĩnh từ chuỗi URL (không cần gọi API ngoài).
Đây là giải pháp cho vấn đề latency của baseline [1].

Tham khảo:
    [1] Shahrivari et al. (2020). Phishing Detection Using Machine Learning
        Techniques. arXiv:2009.11116
        → Cơ sở cho việc chọn các đặc trưng URL.

    [3] Sahoo et al. (2017). Malicious URL Detection using Machine Learning:
        A Survey. arXiv:1701.07179
        → Taxonomy của các đặc trưng URL-based.

Lưu ý: Tất cả các đặc trưng ở đây đều tính toán LOCAL (không cần WHOIS,
DNS lookup, hay HTTP request) để đảm bảo tốc độ inference < 5ms.
"""

import re
import numpy as np
from urllib.parse import urlparse


# Danh sách tên miền shortening phổ biến
SHORTENING_SERVICES = {
    'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly',
    'is.gd', 'buff.ly', 'adf.ly', 'tiny.cc', 'sh.st'
}

# Danh sách TLD đáng ngờ
SUSPICIOUS_TLDS = {
    '.xyz', '.tk', '.ml', '.ga', '.cf', '.gq', '.pw',
    '.top', '.click', '.link', '.online', '.site', '.work'
}


def extract_features(url: str) -> list:
    """
    Trích xuất 15 đặc trưng tĩnh từ 1 URL.

    Args:
        url (str): URL thô, ví dụ "http://paypal-secure.xyz/login"

    Returns:
        list[float]: Vector 15 chiều, sẵn sàng đưa vào model
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()
        full = url.lower()
    except Exception:
        return [0.0] * 15

    features = []

    # --- Nhóm 1: Cấu trúc URL ---
    # F1: Độ dài URL (chuẩn hóa log để tránh outlier)
    features.append(min(np.log1p(len(url)) / 10.0, 1.0))

    # F2: Số lượng dấu chấm trong domain
    features.append(min(domain.count('.') / 5.0, 1.0))

    # F3: Số dấu gạch ngang trong domain (dấu hiệu typosquatting)
    features.append(min(domain.count('-') / 5.0, 1.0))

    # F4: Số thư mục trong path (// nhiều = đáng ngờ)
    features.append(min(path.count('/') / 10.0, 1.0))

    # F5: Có ký tự @ không (browser bỏ qua phần trước @)
    features.append(1.0 if '@' in url else 0.0)

    # F6: Có dùng IP thay domain không (dấu hiệu phishing mạnh)
    ip_pattern = r'(\d{1,3}\.){3}\d{1,3}'
    features.append(1.0 if re.search(ip_pattern, domain) else 0.0)

    # F7: Dùng dịch vụ rút gọn URL
    features.append(1.0 if any(s in domain for s in SHORTENING_SERVICES) else 0.0)

    # F8: HTTPS hay không (1 = có HTTPS, an toàn hơn)
    features.append(1.0 if parsed.scheme == 'https' else 0.0)

    # F9: Có "https" giả trong domain (http://https-paypal.com)
    features.append(1.0 if 'https' in domain else 0.0)

    # F10: Số ký tự đặc biệt trong URL
    special_chars = sum(1 for c in url if c in '!$&\'()*+,;=%~`^{}[]|\\<>')
    features.append(min(special_chars / 20.0, 1.0))

    # --- Nhóm 2: Hành vi đáng ngờ ---
    # F11: Có "login", "signin", "verify", "secure" trong URL
    suspicious_words = ['login', 'signin', 'verify', 'secure', 'account',
                        'update', 'confirm', 'banking', 'paypal', 'ebay']
    features.append(1.0 if any(w in full for w in suspicious_words) else 0.0)

    # F12: Số subdomain (domain.sub.sub.com = 2 subdomain)
    parts = domain.split('.')
    num_subdomains = max(0, len(parts) - 2)
    features.append(min(num_subdomains / 3.0, 1.0))

    # F13: Độ dài domain (domain dài thường là phishing)
    features.append(min(len(domain) / 50.0, 1.0))

    # F14: Có TLD đáng ngờ không
    tld = '.' + domain.split('.')[-1] if '.' in domain else ''
    features.append(1.0 if tld in SUSPICIOUS_TLDS else 0.0)

    # F15: Tỷ lệ chữ số trong URL (nhiều số = đáng ngờ)
    digit_ratio = sum(c.isdigit() for c in url) / max(len(url), 1)
    features.append(min(digit_ratio * 3.0, 1.0))

    return features


def extract_batch(url_list: list) -> np.ndarray:
    """
    Trích xuất features cho toàn bộ danh sách URL.

    Args:
        url_list: List[str]

    Returns:
        np.ndarray shape (N, 15)
    """
    return np.array([extract_features(url) for url in url_list], dtype=np.float32)


# Tên các features để dùng trong báo cáo / visualization
FEATURE_NAMES = [
    'url_length_log',
    'dot_count',
    'hyphen_count',
    'slash_count',
    'has_at_symbol',
    'has_ip_address',
    'is_url_shortener',
    'has_https',
    'fake_https_in_domain',
    'special_char_count',
    'has_suspicious_words',
    'num_subdomains',
    'domain_length',
    'suspicious_tld',
    'digit_ratio',
]
