"""
create_demo_dataset.py
----------------------
Tạo dataset nhỏ (~2000 URL) để test code mà không cần tải ISCX.

QUAN TRỌNG: Dataset này CHỈ để test code chạy được không bị lỗi.
Kết quả model từ dataset này KHÔNG có ý nghĩa học thuật.
Dùng ISCX-URL-2016 cho báo cáo chính thức.

Cách dùng:
    python create_demo_dataset.py
"""

import os
import random
import string
import pandas as pd

random.seed(42)

LEGIT_DOMAINS = [
    'google.com', 'facebook.com', 'youtube.com', 'amazon.com', 'wikipedia.org',
    'twitter.com', 'instagram.com', 'linkedin.com', 'github.com', 'stackoverflow.com',
    'reddit.com', 'microsoft.com', 'apple.com', 'netflix.com', 'spotify.com',
    'coursera.org', 'edx.org', 'medium.com', 'theverge.com', 'bbc.com',
]

PHISHING_PATTERNS = [
    'paypal-secure-{}.xyz',
    'account-verify-{}.tk',
    'secure-login-{}.ml',
    '{}-banking-update.cf',
    'signin-{}-verify.gq',
    '{}.phishing-site.xyz',
    'update-account-{}.tk',
    'confirm-{}-identity.ml',
]

PATHS_LEGIT = [
    '/', '/home', '/about', '/contact', '/products',
    '/search?q=python', '/login', '/dashboard', '/blog/post-1',
    '/docs/getting-started', '/api/v2/users', '/news/technology',
]

PATHS_PHISHING = [
    '/login', '/signin', '/verify', '/account/update',
    '/secure/banking', '/confirm-identity', '/reset-password',
    '/billing/update', '/suspended/verify', '/limited/action',
]


def random_string(length=6):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


def generate_legit_url():
    domain = random.choice(LEGIT_DOMAINS)
    path = random.choice(PATHS_LEGIT)
    scheme = random.choice(['https', 'https', 'https', 'http'])
    return f"{scheme}://www.{domain}{path}"


def generate_phishing_url():
    pattern = random.choice(PHISHING_PATTERNS)
    domain = pattern.format(random_string(random.randint(4, 10)))
    path = random.choice(PATHS_PHISHING)
    scheme = 'http'  # Phishing thường không có https

    # Thêm các dấu hiệu phishing ngẫu nhiên
    tricks = [
        '',
        f'?redirect=https://paypal.com',
        f'@{random_string(8)}.xyz{path}',
        f'//double-slash-trick.com',
    ]
    trick = random.choice(tricks)
    return f"{scheme}://{domain}{path}{trick}"


def main():
    os.makedirs("data/raw", exist_ok=True)

    n_legit = 1000
    n_phishing = 1000

    legit_urls    = [generate_legit_url() for _ in range(n_legit)]
    phishing_urls = [generate_phishing_url() for _ in range(n_phishing)]

    df = pd.DataFrame({
        'url'  : legit_urls + phishing_urls,
        'type' : ['benign'] * n_legit + ['phishing'] * n_phishing
    })
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    out_path = "data/raw/URL_Classification.csv"
    df.to_csv(out_path, index=False)

    print(f"✅ Demo dataset tạo xong: {out_path}")
    print(f"   Legit   : {n_legit} URLs")
    print(f"   Phishing: {n_phishing} URLs")
    print(f"   Total   : {len(df)} URLs")
    print("\n⚠️  Dataset này chỉ để test code.")
    print("   Dùng ISCX-URL-2016 cho kết quả có ý nghĩa học thuật.")


if __name__ == "__main__":
    main()
