# training_pipeline/config_training.py

import os

# --- Cấu hình Đường dẫn ---
# Cách tiếp cận an toàn để xác định thư mục gốc của dự án
# Giả định cấu trúc: project_root/training_pipeline/config_training.py
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Thư mục chứa dữ liệu thô
DATA_DIR = os.path.join(PROJECT_ROOT_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "legal_text_classification.csv")

# Thư mục đích để pipeline huấn luyện lưu các artifact cuối cùng
# API sẽ đọc từ thư mục này
# Quan trọng: Đảm bảo user chạy script này có quyền ghi vào thư mục app/model
TRAINED_ARTIFACTS_DIR = os.path.join(PROJECT_ROOT_DIR, "app", "model", "latest_training_artifacts")

# Thư mục tạm thời để lưu các file plots trước khi log vào MLflow
# Sẽ được tạo và xóa tự động
TEMP_PLOT_DIR = os.path.join(os.path.dirname(__file__), "temp_plots_output")

PIPELINE_FILENAME = "sklearn_pipeline_tfidf_xgboost.joblib"
LABEL_ENCODER_FILENAME = "label_encoder.joblib"

PIPELINE_SAVE_PATH = os.path.join(TRAINED_ARTIFACTS_DIR, PIPELINE_FILENAME)
LABEL_ENCODER_SAVE_PATH = os.path.join(TRAINED_ARTIFACTS_DIR, LABEL_ENCODER_FILENAME)


# --- Cấu hình MLflow ---
# Lấy từ biến môi trường nếu có, nếu không thì dùng giá trị mặc định
# Điều này rất hữu ích khi chạy trong Docker hoặc CI/CD
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT_NAME = "Server_Legal_XGBoost_Pipeline_V4"


# --- Cấu hình Dữ liệu và Tiền xử lý ---
TEXT_COLUMN = "case_title"  
LABEL_COLUMN = "case_outcome"
RANDOM_STATE = 42
TEST_SIZE = 0.2
# Loại tiền xử lý được sử dụng bởi hàm trong app/preprocessing_module.py
# (lemmatization hoặc stemming)
TEXT_PROCESSING_TYPE = "lemmatization" 


# --- Cấu hình Pipeline (TF-IDF & XGBoost) ---
# TF-IDF Vectorizer
MAX_FEATURES_TFIDF = 7500
NGRAM_RANGE_TFIDF = (1, 2)
MIN_DF_TFIDF = 3
MAX_DF_TFIDF = 0.8

# XGBoost Classifier
XGB_PARAMS = {
    'objective': 'multi:softmax',
    'num_class': 10, # Sẽ được cập nhật động trong quá trình chạy
    'eval_metric': 'mlogloss',
    'eta': 0.05,
    'max_depth': 7,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 300,
    'seed': RANDOM_STATE,
    'use_label_encoder': False # Tắt cảnh báo của XGBoost phiên bản cũ
}

# In ra các cấu hình chính để dễ dàng debug
print("--- Config for Training Pipeline loaded ---")
print(f"Project Root: {PROJECT_ROOT_DIR}")
print(f"Artifacts will be saved to: {TRAINED_ARTIFACTS_DIR}")
print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
print("---")