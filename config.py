import os

# --- Đường dẫn và Cấu hình chung ---
DATA_PATH = '/kaggle/input/legal-text-classification-dataset/legal_text_classification.csv'
MODEL_SAVE_DIR = "saved_models_xgboost_joblib" # Đổi tên thư mục output
MLFLOW_EXPERIMENT_NAME = "Kaggle_Legal_XGBoost_Joblib"

# --- Cấu hình Tiền xử lý ---
RANDOM_STATE = 42
TEST_SIZE = 0.2
TEXT_PROCESSING_TYPE = "lemmatization"

# --- Cấu hình Vectorizer (TF-IDF) ---
MAX_FEATURES_TFIDF = 5000
NGRAM_RANGE_TFIDF = (1, 2)
MIN_DF_TFIDF = 5
MAX_DF_TFIDF = 0.7

# --- Cấu hình Mô hình XGBoost ---
XGB_PARAMS = {
    'objective': 'multi:softmax', # Hoặc 'multi:softprob' nếu bạn muốn xác suất
    'num_class': 10,
    'eval_metric': 'mlogloss',
    'eta': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 250,
    'seed': RANDOM_STATE,
    'use_label_encoder': False
}

# Tạo thư mục lưu model nếu chưa có
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
print("Config for XGBoost (Joblib saving) loaded.")
