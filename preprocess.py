import pandas as pd
import re
import nltk
import os
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

import config # Import config đã được viết ra file

def download_nltk_resources():
    nltk_download_dir = "/kaggle/working/nltk_data"
    if not os.path.exists(nltk_download_dir):
        os.makedirs(nltk_download_dir)
    if nltk_download_dir not in nltk.data.path:
        nltk.data.path.append(nltk_download_dir)
    # print("Đang tải các tài nguyên NLTK (sẽ bỏ qua nếu đã có)...") # Bỏ comment nếu muốn thấy log
    nltk.download("wordnet", download_dir=nltk_download_dir, quiet=True)
    nltk.download("punkt", download_dir=nltk_download_dir, quiet=True)
    nltk.download("stopwords", download_dir=nltk_download_dir, quiet=True)
    # print("Tất cả tài nguyên NLTK cần thiết đã sẵn sàng.")

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    marks_and_digits = r'''!()-[]{};?@#$%:'"\\,|./^&;*_0123456789'''
    text = ''.join(char for char in text if char not in marks_and_digits)
    unwanted_phrases = ['url', 'privacy policy', 'disclaimer', 'copyright policy']
    for phrase in unwanted_phrases: text = text.replace(phrase, '')
    return re.sub(r'\s+', ' ', text).strip()

def tokenize_and_process(text, stop_words_list, processor_func, processor_type):
    tokens = text.split() # Sử dụng split() mặc định của Python cho tokenization đơn giản
    processed_tokens = []
    for word in tokens:
        if word not in stop_words_list and len(word) > 2: # Bỏ stop words và từ ngắn
            if processor_type == 'stem':
                processed_tokens.append(processor_func(word))
            elif processor_type == 'lem':
                # WordNetLemmatizer có thể nhận pos tag, 'v' (verb) là một giả định chung
                processed_tokens.append(processor_func(word, pos='v'))
    return processed_tokens

def load_and_preprocess_for_xgboost():
    """Tải và tiền xử lý dữ liệu CHỈ cho XGBoost với TF-IDF."""
    download_nltk_resources()
    print("--- Bắt đầu Tải và Tiền xử lý Dữ liệu cho XGBoost ---")
    try:
        df = pd.read_csv(config.DATA_PATH, on_bad_lines="skip", engine='python')
        print(f"Đã tải {len(df)} dòng từ {config.DATA_PATH}")
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file dữ liệu tại '{config.DATA_PATH}'")
        return None, None, None # Trả về 3 giá trị để phù hợp với hàm gọi

    df = df.fillna('') # Xử lý NaN
    df['case_text_sum'] = df['case_title'] + " " + df['case_text']
    df['clean_text'] = df['case_text_sum'].apply(clean_text)

    # Mã hóa nhãn
    le = LabelEncoder()
    df['case_outcome_num'] = le.fit_transform(df['case_outcome'])
    num_classes = len(le.classes_)
    print(f"Số lượng lớp (num_classes): {num_classes}")
    # Cập nhật num_class trong XGB_PARAMS nếu chưa khớp (dù sẽ được cập nhật lại trong train)
    config.XGB_PARAMS['num_class'] = num_classes


    # Tiền xử lý văn bản (stemming hoặc lemmatization)
    stop_words = nltk_stopwords.words('english')
    if config.TEXT_PROCESSING_TYPE == "stemming":
        print("Đang xử lý Stemming...")
        processor = PorterStemmer()
        processor_type = 'stem'
        df['tokens'] = df['clean_text'].apply(lambda x: tokenize_and_process(x, stop_words, processor.stem, processor_type))
    elif config.TEXT_PROCESSING_TYPE == "lemmatization":
        print("Đang xử lý Lemmatization...")
        processor = WordNetLemmatizer()
        processor_type = 'lem'
        df['tokens'] = df['clean_text'].apply(lambda x: tokenize_and_process(x, stop_words, processor.lemmatize, processor_type))
    else:
        raise ValueError(f"Loại tiền xử lý không hợp lệ: {config.TEXT_PROCESSING_TYPE}. Chọn 'stemming' hoặc 'lemmatization'.")

    df['processed_text_joined'] = df['tokens'].apply(' '.join)
    print("Đã hoàn tất tiền xử lý văn bản.")
    return df, le, num_classes

def get_tfidf_datasets_for_xgboost(df, le):
    """Tạo dataset TF-IDF cho XGBoost."""
    print("--- Tạo Dataset TF-IDF cho XGBoost ---")
    y_encoded = df['case_outcome_num'] # Nhãn đã được mã hóa
    X_processed_text = df['processed_text_joined']

    # Phân chia dữ liệu
    X_train_text, X_test_text, y_train_encoded, y_test_encoded = train_test_split(
        X_processed_text, y_encoded,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y_encoded # Quan trọng để giữ tỷ lệ lớp
    )

    # Vector hóa TF-IDF
    tfidf_vectorizer = TfidfVectorizer(
        max_features=config.MAX_FEATURES_TFIDF,
        ngram_range=config.NGRAM_RANGE_TFIDF,
        min_df=config.MIN_DF_TFIDF,
        max_df=config.MAX_DF_TFIDF
    )

    print("Đang fit và transform TF-IDF cho tập huấn luyện...")
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
    print("Đang transform TF-IDF cho tập kiểm thử...")
    X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

    print(f"Kích thước X_train_tfidf: {X_train_tfidf.shape}")
    print(f"Kích thước X_test_tfidf: {X_test_tfidf.shape}")

    return X_train_tfidf, X_test_tfidf, y_train_encoded, y_test_encoded, tfidf_vectorizer
