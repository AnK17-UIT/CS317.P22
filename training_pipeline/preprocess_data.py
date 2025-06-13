# -*- coding: utf-8 -*-
# training_pipeline/preprocess_data.py
import pandas as pd
import re
import nltk
import os
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.preprocessing import LabelEncoder

# Import config từ cùng thư mục
try:
    import config_training as config
except ImportError:
    print("LỖI: Không thể import config_training.py. Đảm bảo nó cùng cấp với script này.")
    raise

# Import hàm tiền xử lý từ app/preprocessing_module.py
# Cần thêm đường dẫn đến thư mục gốc của dự án vào sys.path
import sys
PROJECT_ROOT_ABS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT_ABS_PATH)
try:
    from app.preprocessing_module import preprocess_text_for_prediction_api as text_processor_func
    from app.preprocessing_module import download_nltk_resources_for_api
except ImportError:
    print("LỖI: Không thể import từ app.preprocessing_module. Kiểm tra sys.path và cấu trúc thư mục.")
    raise

def load_and_preprocess_data_for_server_training():
    """Tải và tiền xử lý dữ liệu cho việc huấn luyện trên server."""
    download_nltk_resources_for_api() # Đảm bảo tài nguyên NLTK được tải
    print("--- Bước 1: Bắt đầu Tải và Tiền xử lý Dữ liệu trên Server ---")

    try:
        df = pd.read_csv(config.RAW_DATA_PATH, on_bad_lines="skip", engine='python')
        print(f"Đã tải {len(df)} dòng từ {config.RAW_DATA_PATH}")
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file dữ liệu tại '{config.RAW_DATA_PATH}'")
        return None, None, None
    except Exception as e:
        print(f"LỖI khi tải dữ liệu: {e}")
        return None, None, None

    df = df.fillna('')
    # Kiểm tra xem các cột cần thiết có tồn tại không
    if 'case_title' not in df.columns or 'case_text' not in df.columns or config.LABEL_COLUMN not in df.columns:
        print(f"LỖI: Thiếu một trong các cột cần thiết: 'case_title', 'case_text', '{config.LABEL_COLUMN}'")
        return None, None, None

    df['case_text_sum'] = df['case_title'] + " " + df['case_text']

    # Sử dụng hàm tiền xử lý đã định nghĩa trong app/preprocessing_module.py
    # để đảm bảo tính nhất quán giữa huấn luyện và dự đoán của API
    print("Đang tiền xử lý văn bản (sử dụng hàm từ app.preprocessing_module)...")
    df['processed_text_joined'] = df['case_text_sum'].apply(text_processor_func)
    print("Tiền xử lý văn bản hoàn tất.")

    # Loại bỏ các hàng có processed_text rỗng
    df = df[df['processed_text_joined'].str.strip().astype(bool)]
    if df.empty:
        print("LỖI: Không còn dữ liệu sau khi loại bỏ các văn bản rỗng đã xử lý.")
        return None, None, None
    print(f"Còn lại {len(df)} dòng sau khi loại bỏ văn bản rỗng đã xử lý.")

    # Mã hóa nhãn
    le = LabelEncoder()
    try:
        df['case_outcome_num'] = le.fit_transform(df[config.LABEL_COLUMN])
    except Exception as e:
        print(f"LỖI khi mã hóa nhãn: {e}")
        return None, None, None

    num_classes = len(le.classes_)
    print(f"Số lượng lớp (num_classes): {num_classes}")
    config.XGB_PARAMS['num_class'] = num_classes # Cập nhật config động

    print("--- Bước 1: Hoàn tất Tải và Tiền xử lý Dữ liệu ---")
    return df, le, num_classes

if __name__ == "__main__":
    # Chỉ để kiểm tra script này
    df_processed, label_enc, num_cls = load_and_preprocess_data_for_server_training()
    if df_processed is not None:
        print("\nKiểm tra dữ liệu đã xử lý:")
        print(df_processed[['processed_text_joined', 'case_outcome_num']].head())
        print(f"\nLabel Encoder Classes: {list(label_enc.classes_)}")
        print(f"Number of classes: {num_cls}")