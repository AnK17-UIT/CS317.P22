# training_pipeline/train_model_pipeline.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import mlflow
import mlflow.sklearn
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

# --- Thiết lập đường dẫn và Imports ---
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_DIR)

try:
    import config_training as config
    from preprocess_data import load_and_preprocess_data
except ImportError as e:
    print(f"FATAL: Không thể import các module cần thiết. Lỗi: {e}")
    raise

def log_artifacts_to_mlflow(y_true, y_pred, le):
    """Tính toán metrics và log tất cả các artifacts (report, plot) vào MLflow."""
    print("INFO: Đang tính toán và log metrics, artifacts...")
    
    try:
        # 1. Log Classification Report
        report_text = classification_report(y_true, y_pred, target_names=le.classes_)
        mlflow.log_text(report_text, "classification_report.txt")
        print("  - Đã log classification_report.txt")

        # 2. Tạo và Log Confusion Matrix
        # Tạo thư mục tạm để lưu ảnh
        os.makedirs(config.TEMP_PLOT_DIR, exist_ok=True)
        cm_path = os.path.join(config.TEMP_PLOT_DIR, 'confusion_matrix.png')
        
        cm = confusion_matrix(y_true, y_pred, labels=range(len(le.classes_)))
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close() # Rất quan trọng để giải phóng bộ nhớ
        
        mlflow.log_artifact(cm_path, "plots")
        print("  - Đã log confusion_matrix.png")

    except Exception as e:
        print(f"CẢNH BÁO: Đã xảy ra lỗi khi tạo hoặc log artifacts: {e}")
    finally:
        # Dọn dẹp thư mục tạm bất kể thành công hay thất bại
        if os.path.exists(config.TEMP_PLOT_DIR):
            shutil.rmtree(config.TEMP_PLOT_DIR)
            print(f"  - Đã dọn dẹp thư mục tạm: {config.TEMP_PLOT_DIR}")

def run_training_pipeline():
    """Hàm chính điều phối toàn bộ quy trình huấn luyện và lưu trữ."""
    
    # --- Bước 1: Chuẩn bị dữ liệu ---
    df, le = load_and_preprocess_data()
    if df is None or le is None:
        print("FATAL: Dữ liệu không được chuẩn bị thành công. Dừng pipeline.")
        return False

    # --- Bước 2: Phân chia dữ liệu ---
    X = df['processed_text']
    y = df['case_outcome_num']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    print(f"\n--- BƯỚC 2: ĐÃ PHÂN CHIA DỮ LIỆU ---")
    print(f"Tập huấn luyện: {len(X_train)} mẫu | Tập kiểm thử: {len(X_test)} mẫu")

    # --- Bước 3: Thiết lập và Chạy MLflow ---
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    
    pipeline_name_suffix = config.TEXT_PROCESSING_TYPE.capitalize()
    run_name = f"Server_SklearnPipeline_XGBoost_TFIDF_{pipeline_name_suffix}"

    with mlflow.start_run(run_name=run_name) as run:
        print(f"\n--- BƯỚC 3: BẮT ĐẦU MLFLOW RUN: '{run_name}' (ID: {run.info.run_id}) ---")
        
        try:
            # 3.1. Log các tham số
            mlflow.log_params(config.XGB_PARAMS)
            mlflow.log_params({
                "tfidf_max_features": config.MAX_FEATURES_TFIDF,
                "tfidf_ngram_range": str(config.NGRAM_RANGE_TFIDF),
                "tfidf_min_df": config.MIN_DF_TFIDF,
                "tfidf_max_df": config.MAX_DF_TFIDF,
                "text_processing_type": config.TEXT_PROCESSING_TYPE,
            })
            print("INFO: Đã log các tham số vào MLflow.")

            # 3.2. Định nghĩa và Huấn luyện Pipeline
            training_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=config.MAX_FEATURES_TFIDF, ngram_range=config.NGRAM_RANGE_TFIDF,
                    min_df=config.MIN_DF_TFIDF, max_df=config.MAX_DF_TFIDF
                )),
                ('xgb', XGBClassifier(**config.XGB_PARAMS))
            ])
            
            print("INFO: Đang huấn luyện pipeline...")
            training_pipeline.fit(X_train, y_train)
            print("INFO: Huấn luyện hoàn tất.")

            # 3.3. Đánh giá và Log kết quả
            y_pred = training_pipeline.predict(X_test)
            log_artifacts_to_mlflow(y_test, y_pred, le)

            # 3.4. Lưu Model và các thành phần
            print("INFO: Đang lưu pipeline và label encoder...")
            # Log vào MLflow
            mlflow.sklearn.log_model(
                sk_model=training_pipeline,
                artifact_path="model_pipeline",
                registered_model_name="LegalCaseClassifierXGBoost"
            )
            print("  - Đã log model pipeline vào MLflow.")
            
            # Lưu vào thư mục cục bộ để API sử dụng
            # Đảm bảo thư mục đích tồn tại
            os.makedirs(config.TRAINED_ARTIFACTS_DIR, exist_ok=True)
            joblib.dump(training_pipeline, config.PIPELINE_SAVE_PATH)
            joblib.dump(le, config.LABEL_ENCODER_SAVE_PATH)
            print(f"  - Pipeline và Label Encoder đã được lưu vào: {config.TRAINED_ARTIFACTS_DIR}")
            
            mlflow.set_tag("status", "SUCCESS")

        except Exception as e:
            mlflow.set_tag("status", "FAILED")
            print(f"FATAL: Đã xảy ra lỗi nghiêm trọng trong MLflow run: {e}")
            import traceback
            traceback.print_exc()
            return False

    print(f"--- ĐÃ HOÀN THÀNH MLFLOW RUN: '{run_name}' ---")
    return True

if __name__ == "__main__":
    print(">>> BẮT ĐẦU CHẠY PIPELINE HUẤN LUYỆN ĐỘC LẬP <<<")
    success = run_training_pipeline()
    if success:
        print("\n>>> PIPELINE HUẤN LUYỆN HOÀN THÀNH THÀNH CÔNG <<<")
    else:
        print("\n>>> PIPELINE HUẤN LUYỆN ĐÃ THẤT BẠI <<<")
        sys.exit(1)