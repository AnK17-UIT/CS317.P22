# -*- coding: utf-8 -*-
# training_pipeline/train_model_pipeline.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score # Dòng này có thể là dòng 10 hoặc gần đó
import joblib
import mlflow
import mlflow.sklearn # Để log Sklearn pipeline
import os
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import config_training as config
except ImportError:
    print("LỖI: Không thể import config_training.py.")
    raise

def compute_and_log_metrics_for_server_pipeline(y_true_encoded, y_pred_encoded, pipeline_name_suffix, le, num_classes):
    """Tính toán, in và log các metrics cho MLflow cho một pipeline huấn luyện trên server."""
    y_true_labels = le.inverse_transform(y_true_encoded)
    y_pred_labels = le.inverse_transform(y_pred_encoded)

    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    f1 = f1_score(y_true_labels, y_pred_labels, average='macro')
    
    run_name_full = "Server_SklearnPipeline_XGBoost_TFIDF_{}".format(pipeline_name_suffix)
    
    print("  Kết quả cho {}:".format(run_name_full))
    print("    Accuracy: {:.3f}%".format(accuracy * 100))
    print("    F1 Score (Macro): {:.3f}%".format(f1 * 100))

    mlflow.log_metrics({
        "test_accuracy_{}".format(pipeline_name_suffix.lower()): accuracy,
        "test_f1_macro_{}".format(pipeline_name_suffix.lower()): f1
    })

    # Confusion Matrix
    cm_path_local = None # Khởi tạo để tránh lỗi nếu plot không thành công
    try:
        cm = confusion_matrix(y_true_labels, y_pred_labels, labels=le.classes_)
        # Đảm bảo num_classes là int khi dùng trong phép chia //
        fig_width = max(8, int(num_classes // 1.5))
        fig_height = max(6, int(num_classes // 2))
        plt.figure(figsize=(fig_width, fig_height))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title('Confusion Matrix - {}'.format(run_name_full))
        plt.xlabel('Predicted Label'); plt.ylabel('True Label')
        plt.tight_layout()
        cm_filename = "confusion_matrix_{}.png".format(run_name_full.replace(' ', '_'))
        cm_path_local = cm_filename
        plt.savefig(cm_path_local)
        mlflow.log_artifact(cm_path_local, "plots") # Log vào thư mục con 'plots' trong artifacts
        plt.close()
        print("    Confusion matrix đã được log vào MLflow artifacts/plots/{}".format(cm_filename))
    except Exception as e:
        print("LỖI khi tạo hoặc log confusion matrix: {}".format(e))
    finally:
        if cm_path_local and os.path.exists(cm_path_local): # Xóa file tạm sau khi log
             os.remove(cm_path_local)


def train_and_save_pipeline_on_server(df_processed, le, num_classes):
    """Huấn luyện và lưu Sklearn Pipeline (TF-IDF + XGBoost) và LabelEncoder."""
    print("\n--- Bước 2: Bắt đầu Huấn luyện và Lưu trữ Pipeline trên Server ---")

    if df_processed is None or df_processed.empty or le is None:
        print("LỖI: Dữ liệu đầu vào không hợp lệ cho việc huấn luyện. Dừng lại.")
        return False

    X_text_processed = df_processed['processed_text_joined']
    y_encoded = df_processed['case_outcome_num']

    X_train_text, X_test_text, y_train_encoded, y_test_encoded = train_test_split(
        X_text_processed, y_encoded,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y_encoded
    )
    print("Kích thước tập huấn luyện (text): {}".format(len(X_train_text)))
    print("Kích thước tập kiểm thử (text): {}".format(len(X_test_text)))

    # Tạo các bước cho pipeline
    tfidf_step = ('tfidf', TfidfVectorizer(
        max_features=config.MAX_FEATURES_TFIDF,
        ngram_range=config.NGRAM_RANGE_TFIDF,
        min_df=config.MIN_DF_TFIDF,
        max_df=config.MAX_DF_TFIDF
    ))

    current_xgb_params = config.XGB_PARAMS.copy()
    # num_classes đã được cập nhật trong config bởi preprocess_data.py
    # current_xgb_params['num_class'] = num_classes # Đảm bảo lại lần nữa
    
    xgboost_step = ('xgb', XGBClassifier(**current_xgb_params))

    pipeline = Pipeline([tfidf_step, xgboost_step])

    # Cấu hình MLflow
    if config.MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    pipeline_suffix = config.TEXT_PROCESSING_TYPE.capitalize()
    run_name = "Server_SklearnPipeline_XGBoost_TFIDF_{}".format(pipeline_suffix)

    with mlflow.start_run(run_name=run_name) as run:
        print("\n[MLflow Run] Bắt đầu run: {} (ID: {})".format(run_name, run.info.run_id))
        
        mlflow.log_param("text_processing_type", config.TEXT_PROCESSING_TYPE)
        mlflow.log_params(config.XGB_PARAMS)
        mlflow.log_param("tfidf_max_features", config.MAX_FEATURES_TFIDF)
        mlflow.log_param("tfidf_ngram_range", str(config.NGRAM_RANGE_TFIDF))
        mlflow.log_param("tfidf_min_df", config.MIN_DF_TFIDF)
        mlflow.log_param("tfidf_max_df", config.MAX_DF_TFIDF)


        print("Đang huấn luyện Pipeline (TF-IDF -> XGBoost)...")
        pipeline.fit(X_train_text, y_train_encoded)
        print("Huấn luyện Pipeline hoàn tất.")

        y_pred_encoded = pipeline.predict(X_test_text)
        compute_and_log_metrics_for_server_pipeline(y_test_encoded, y_pred_encoded, pipeline_suffix, le, num_classes)

        print("Đang log Pipeline vào MLflow và lưu cục bộ...")
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="sklearn_pipeline_tfidf_xgboost",
        )
        try:
            joblib.dump(pipeline, config.PIPELINE_SAVE_PATH)
            print("Pipeline đã được lưu cục bộ tại: {}".format(config.PIPELINE_SAVE_PATH))
            joblib.dump(le, config.LABEL_ENCODER_SAVE_PATH)
            print("Label Encoder đã được lưu cục bộ tại: {}".format(config.LABEL_ENCODER_SAVE_PATH))
        except Exception as e: # Sửa lại biến lỗi
            print("LỖI khi lưu pipeline/encoder cục bộ: {}".format(e))
            mlflow.log_param("local_save_status", "FAILED")
            return False

        mlflow.log_param("local_save_status", "SUCCESS")
        print("Đã hoàn tất run MLflow: {}".format(run_name))

    print("--- Bước 2: Hoàn tất Huấn luyện và Lưu trữ Pipeline ---")
    return True

if __name__ == "__main__":
    print("Chạy thử train_model_pipeline.py (cần dữ liệu đã xử lý)...")
    # Bước 1: Gọi hàm tiền xử lý để lấy df_processed, le, num_classes
    # Giả định preprocess_data.py có thể được import và hàm của nó có thể được gọi
    # Điều này cần preprocess_data.py cũng nằm trong training_pipeline/ hoặc sys.path được cấu hình đúng
    try:
        # Thêm sys.path nếu preprocess_data.py không cùng cấp và không trong PYTHONPATH
        import sys
        import os
        # Giả định preprocess_data.py nằm cùng cấp với train_model_pipeline.py
        # Nếu không, bạn cần điều chỉnh sys.path cho phù hợp
        # Ví dụ, nếu cả hai đều trong training_pipeline:
        # SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        # sys.path.append(SCRIPT_DIR) # Đảm bảo có thể import các file cùng thư mục

        from preprocess_data import load_and_preprocess_data_for_server_training
        df_processed_test, le_test, num_classes_test = load_and_preprocess_data_for_server_training()
        
        if df_processed_test is not None:
            train_and_save_pipeline_on_server(df_processed_test, le_test, num_classes_test)
        else:
            print("Không có dữ liệu đã xử lý để chạy thử huấn luyện.")
    except ImportError as e:
        print("LỖI ImportError khi cố gắng chạy __main__ của train_model_pipeline.py: {}".format(e))
        print("Đảm bảo preprocess_data.py có thể được import.")
    except Exception as e:
        print("LỖI không xác định khi chạy __main__ của train_model_pipeline.py: {}".format(e))