import mlflow
import os
import joblib # Sử dụng joblib để lưu model
import shutil

# Import các module đã viết ra file
import config_training
from preprocess_data import load_and_preprocess_for_xgboost
from training_pipeline.train_model_pipeline import train_xgboost_model_only

def save_xgboost_artifacts_joblib(run_info, le): # Đổi tên hàm
    """Lưu mô hình XGBoost tốt nhất và các artifact cần thiết bằng joblib."""
    print("\n" + "="*50)
    print("--- BẮT ĐẦU LƯU ARTIFACTS CỦA XGBOOST (JOBLIB) CHO TRIỂN KHAI ---")

    run_id = run_info.info.run_id
    model_name_from_tag = run_info.data.tags.get('mlflow.runName', 'XGBoost_Run_Unknown')
    # Lấy f1 score từ MLflow metrics, đảm bảo tên metric khớp với tên đã log
    metric_key_f1 = f'test_f1_macro_{config_training.TEXT_PROCESSING_TYPE.lower()}' # Sửa lỗi chữ hoa
    f1_score = run_info.data.metrics.get(metric_key_f1, 0)


    print(f"Thông tin Run được chọn:")
    print(f"  Run ID: {run_id}")
    print(f"  Tên Run (Model): {model_name_from_tag}")
    print(f"  F1 Score (Macro) ({metric_key_f1}): {f1_score:.4f}") # In key để dễ debug
    print("="*50 + "\n")

    # Tạo lại thư mục lưu trữ
    if os.path.exists(config_training.MODEL_SAVE_DIR):
        shutil.rmtree(config_training.MODEL_SAVE_DIR)
    os.makedirs(config_training.MODEL_SAVE_DIR)
    print(f"Đã tạo thư mục lưu trữ: {config_training.MODEL_SAVE_DIR}")

    # 1. Lưu Label Encoder
    le_path = os.path.join(config_training.MODEL_SAVE_DIR, "label_encoder.joblib")
    joblib.dump(le, le_path)
    print(f"Đã lưu Label Encoder vào: {le_path}")

    # 2. Tải mô hình XGBoost và Vectorizer TF-IDF từ MLflow artifacts
    print("Đang tải mô hình XGBoost và Vectorizer từ MLflow artifacts...")
    try:
        loaded_xgboost_model = mlflow.xgboost.load_model(f"runs:/{run_id}/xgboost_model")
        loaded_tfidf_vectorizer = mlflow.sklearn.load_model(f"runs:/{run_id}/tfidf_vectorizer")
        print("Tải mô hình và vectorizer từ MLflow thành công.")
    except Exception as e:
        print(f"LỖI khi tải mô hình/vectorizer từ MLflow: {e}")
        return

    # 3. Lưu Vectorizer TF-IDF bằng joblib
    vectorizer_path = os.path.join(config_training.MODEL_SAVE_DIR, "tfidf_vectorizer.joblib")
    joblib.dump(loaded_tfidf_vectorizer, vectorizer_path)
    print(f"Đã lưu Vectorizer TF-IDF vào: {vectorizer_path}")

    # 4. Lưu mô hình XGBoost bằng joblib
    # Đặt tên file rõ ràng là model XGBoost
    xgboost_model_path = os.path.join(config_training.MODEL_SAVE_DIR, "xgboost_model.joblib")
    try:
        joblib.dump(loaded_xgboost_model, xgboost_model_path)
        print(f"Đã lưu mô hình XGBoost (dạng joblib) tại: {xgboost_model_path}")
    except Exception as e:
        print(f"LỖI khi lưu mô hình XGBoost bằng joblib: {e}")

    print("\n--- LƯU ARTIFACTS (JOBLIB) HOÀN TẤT ---")


def main_xgboost_pipeline_joblib(): # Đổi tên hàm chính
    """Hàm chính điều phối pipeline CHỈ cho XGBoost, lưu bằng joblib."""
    mlflow.set_experiment(config_training.MLFLOW_EXPERIMENT_NAME)
    print(f"MLflow experiment được đặt thành: '{config_training.MLFLOW_EXPERIMENT_NAME}'")

    df_processed, le, num_classes = load_and_preprocess_for_xgboost()
    if df_processed is None or le is None:
        print("Không thể tải hoặc tiền xử lý dữ liệu. Pipeline dừng lại.")
        return

    print("\n--- Bắt đầu Pipeline Execution Run cho MLflow ---")
    with mlflow.start_run(run_name="XGBoost_Pipeline_Joblib_Execution") as parent_run: # Đổi tên parent run
        parent_run_id = parent_run.info.run_id
        print(f"Parent Run ID: {parent_run_id}")
        
        train_xgboost_model_only(df_processed, le, num_classes) # Hàm này tạo nested run
        
        mlflow.log_param("parent_run_status", "COMPLETED_JOBLIB")

    print("\n--- Tìm kiếm run XGBoost vừa thực hiện để lưu artifacts (joblib) ---")
    try:
        expected_run_name_part = f"XGBoost_TFIDF_{config_training.TEXT_PROCESSING_TYPE.capitalize()}"
        experiment = mlflow.get_experiment_by_name(config_training.MLFLOW_EXPERIMENT_NAME)
        if experiment is None:
            raise Exception(f"Không tìm thấy experiment: {config_training.MLFLOW_EXPERIMENT_NAME}")

        xgboost_runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}' AND tags.mlflow.runName LIKE '%{expected_run_name_part}%'",
            order_by=["start_time DESC"]
        )

        if xgboost_runs.empty:
            raise Exception(f"Không tìm thấy run XGBoost nào khớp với parent ID {parent_run_id} và tên chứa '{expected_run_name_part}'.")

        best_xgboost_run_info = xgboost_runs.iloc[0]
        print(f"Đã tìm thấy run XGBoost: {best_xgboost_run_info['tags.mlflow.runName']} (ID: {best_xgboost_run_info.run_id})")
        
        xgboost_run_to_save = mlflow.get_run(best_xgboost_run_info.run_id)
        
        save_xgboost_artifacts_joblib(xgboost_run_to_save, le) # Gọi hàm lưu joblib

    except Exception as e:
        print(f"Đã xảy ra lỗi khi tìm kiếm hoặc lưu artifacts của XGBoost (joblib): {e}")

    print("\n--- PIPELINE XGBOOST (JOBLIB) ĐÃ HOÀN TẤT! ---")
    print(f"Model XGBoost và các thành phần đã được lưu (dạng joblib) vào thư mục '{config_training.MODEL_SAVE_DIR}'.")
    print("Sẵn sàng cho việc đóng gói API và Docker.")

# Chạy pipeline
if __name__ == "__main__":
    main_xgboost_pipeline_joblib()