import mlflow
import mlflow.xgboost # Sử dụng mlflow.xgboost
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

import config # Import config

def compute_and_log_metrics(y_true_encoded, y_pred_encoded, model_name_suffix, le, num_classes):
    """Tính toán, in và log các metrics cho MLflow."""
    # Chuyển về nhãn gốc để báo cáo dễ hiểu hơn
    y_true_labels = le.inverse_transform(y_true_encoded)
    y_pred_labels = le.inverse_transform(y_pred_encoded)

    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    f1 = f1_score(y_true_labels, y_pred_labels, average='macro') # macro F1 cho đa lớp
    
    run_name = f"XGBoost_TFIDF_{model_name_suffix}"
    print(f"  Kết quả cho {run_name}:")
    print(f"    Accuracy: {accuracy*100:.3f}%")
    print(f"    F1 Score (Macro): {f1*100:.3f}%")

    mlflow.log_metrics({
        f"test_accuracy_{model_name_suffix.lower()}": accuracy,
        f"test_f1_macro_{model_name_suffix.lower()}": f1
    })

    # Confusion Matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=le.classes_)
    plt.figure(figsize=(max(8, num_classes // 1.5), max(6, num_classes // 2)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix - {run_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    cm_path = f"confusion_matrix_{run_name.replace(' ', '_')}.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path, "plots")
    plt.close() # Đóng plot để tránh hiển thị trong notebook nếu không muốn

def train_xgboost_model_only(df_processed, le, num_classes):
    """Huấn luyện và đánh giá CHỈ mô hình XGBoost."""
    from preprocess import get_tfidf_datasets_for_xgboost # Import hàm tạo dataset

    print("\n" + "="*30)
    print("--- Bắt đầu Huấn luyện Mô hình XGBoost ---")
    print("="*30)

    X_train_tfidf, X_test_tfidf, y_train_encoded, y_test_encoded, tfidf_vectorizer = \
        get_tfidf_datasets_for_xgboost(df_processed, le)

    # Cập nhật num_class trong XGB_PARAMS một lần nữa để chắc chắn
    # (mặc dù đã làm trong preprocess, nhưng đây là nơi nó thực sự được dùng)
    current_xgb_params = config.XGB_PARAMS.copy()
    current_xgb_params['num_class'] = num_classes

    model_suffix = config.TEXT_PROCESSING_TYPE.capitalize() # Sẽ là "Lemmatization" hoặc "Stemming"
    run_name = f"XGBoost_TFIDF_{model_suffix}"

    with mlflow.start_run(run_name=run_name, nested=True) as run:
        print(f"\n[MLflow Run] Bắt đầu run: {run_name} (ID: {run.info.run_id})")
        
        # Log các tham số
        mlflow.log_params(current_xgb_params)
        mlflow.log_param("text_processing", config.TEXT_PROCESSING_TYPE)
        mlflow.log_param("tfidf_max_features", config.MAX_FEATURES_TFIDF)
        mlflow.log_param("tfidf_ngram_range", str(config.NGRAM_RANGE_TFIDF)) # MLflow thích string
        mlflow.log_param("tfidf_min_df", config.MIN_DF_TFIDF)
        mlflow.log_param("tfidf_max_df", config.MAX_DF_TFIDF)


        print("Đang huấn luyện XGBoost...")
        xgb_model = XGBClassifier(**current_xgb_params)
        xgb_model.fit(X_train_tfidf, y_train_encoded)
        print("Huấn luyện XGBoost hoàn tất.")

        # Dự đoán và đánh giá
        y_pred_encoded = xgb_model.predict(X_test_tfidf)
        compute_and_log_metrics(y_test_encoded, y_pred_encoded, model_suffix, le, num_classes)

        # Log mô hình và vectorizer
        print("Đang log mô hình XGBoost và vectorizer TF-IDF vào MLflow...")
        mlflow.xgboost.log_model(xgb_model, "xgboost_model") # Đặt tên artifact path là "xgboost_model"
        mlflow.sklearn.log_model(tfidf_vectorizer, "tfidf_vectorizer") # Đặt tên artifact path là "tfidf_vectorizer"
        
        print(f"Đã hoàn tất run: {run_name}")
    
    print("--- Huấn luyện Mô hình XGBoost Hoàn tất ---")
    # Trả về các thành phần đã fit để có thể lưu nếu cần (dù MLflow đã lưu)
    return xgb_model, tfidf_vectorizer
