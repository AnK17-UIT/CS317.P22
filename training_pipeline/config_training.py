# -*- coding: utf-8 -*-
# training_pipeline/config_training.py
import os

# --- Đường dẫn và Cấu hình chung cho Huấn luyện ---
BASE_TRAINING_DIR = os.path.dirname(os.path.abspath(__file__)) # Thư mục training_pipeline/
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(BASE_TRAINING_DIR, "..")) # Thư mục gốc dự án

# Đường dẫn dữ liệu thô
DATA_DIR = os.path.join(PROJECT_ROOT_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "your_legal_data.csv") # << THAY THẾ TÊN FILE CSV CỦA BẠN

# Thư mục lưu các model artifacts sau khi huấn luyện (sẽ được copy vào app/model/)
# Đây là thư mục tạm thời trong quá trình huấn luyện, hoặc có thể trỏ thẳng đến app/model
TRAINED_ARTIFACTS_OUTPUT_DIR = os.path.join(PROJECT_ROOT_DIR, "app", "model", "latest_training_artifacts")

# Tên file cho Pipeline và LabelEncoder
PIPELINE_FILENAME = "sklearn_pipeline_tfidf_xgboost.joblib"
LABEL_ENCODER_FILENAME = "label_encoder.joblib"

PIPELINE_SAVE_PATH = os.path.join(TRAINED_ARTIFACTS_OUTPUT_DIR, PIPELINE_FILENAME)
LABEL_ENCODER_SAVE_PATH = os.path.join(TRAINED_ARTIFACTS_OUTPUT_DIR, LABEL_ENCODER_FILENAME)

# Cấu hình MLflow (nếu bạn chạy MLflow server trên cùng máy hoặc server khác)
MLFLOW_TRACKING_URI = "http://localhost:5000" # Hoặc URI của MLflow server của bạn
MLFLOW_EXPERIMENT_NAME = "Server_Legal_XGBoost_Pipeline"

# --- Cấu hình Tiền xử lý ---
RANDOM_STATE = 42
TEST_SIZE = 0.2
TEXT_PROCESSING_TYPE = "lemmatization" # "stemming" hoặc "lemmatization"

# --- Cấu hình Vectorizer (TF-IDF) ---
MAX_FEATURES_TFIDF = 5000
NGRAM_RANGE_TFIDF = (1, 2)
MIN_DF_TFIDF = 5
MAX_DF_TFIDF = 0.7

# --- Cấu hình Mô hình XGBoost ---
XGB_PARAMS = {
    'objective': 'multi:softmax',
    'num_class': 10, # Sẽ được cập nhật
    'eval_metric': 'mlogloss',
    'eta': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 250,
    'seed': RANDOM_STATE,
    'use_label_encoder': False
}

# Tạo các thư mục cần thiết nếu chưa có
os.makedirs(TRAINED_ARTIFACTS_OUTPUT_DIR, exist_ok=True)
# os.makedirs(os.path.join(PROJECT_ROOT_DIR, "app", "model"), exist_ok=True) # Đảm bảo app/model tồn tại

print("Config for Server Training loaded.")
print(f"Project Root: {PROJECT_ROOT_DIR}")
print(f"Raw Data Path: {RAW_DATA_PATH}")
print(f"Trained Artifacts Output Dir: {TRAINED_ARTIFACTS_OUTPUT_DIR}")