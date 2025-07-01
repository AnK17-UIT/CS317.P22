#!/bin/bash

echo "===== BẮT ĐẦU PIPELINE HUẤN LUYỆN MÔ HÌNH TRÊN SERVER UBUNTU ====="
set -e

TRAINING_PIPELINE_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT_DIR=$(realpath "$TRAINING_PIPELINE_DIR/..")

echo "Thư mục gốc dự án: $PROJECT_ROOT_DIR"
echo "Thư mục pipeline huấn luyện: $TRAINING_PIPELINE_DIR"

VENV_PATH="$PROJECT_ROOT_DIR/my_project_env" # Tên môi trường ảo của bạn

if [ -f "$VENV_PATH/bin/activate" ]; then
  echo "Kích hoạt môi trường ảo tại $VENV_PATH..."
  source "$VENV_PATH/bin/activate"
  echo "PYTHONPATH hiện tại (sau khi kích hoạt venv): '$PYTHONPATH'"
  echo "Đường dẫn thực thi Python (which python):"
  which python
  echo "Phiên bản Python (python --version):"
  python --version
else
  echo "CẢNH BÁO: Không tìm thấy file activate của môi trường ảo tại '$VENV_PATH/bin/activate'."
  echo "Vui lòng tạo và cài đặt các gói vào môi trường ảo trước khi chạy script này."
  exit 1
fi

mkdir -p "$PROJECT_ROOT_DIR/app/model/latest_training_artifacts"
echo "Đảm bảo thư mục output cho artifacts tồn tại: $PROJECT_ROOT_DIR/app/model/latest_training_artifacts"

TRAIN_SCRIPT="$TRAINING_PIPELINE_DIR/train_model_pipeline.py"

echo "\n>>> Chạy Bước Chính: Huấn luyện và Lưu trữ Pipeline..."
python "$TRAIN_SCRIPT"
SCRIPT_EXIT_CODE=$?

if [ $SCRIPT_EXIT_CODE -ne 0 ]; then
    echo "LỖI: Bước huấn luyện và lưu trữ pipeline thất bại với mã thoát $SCRIPT_EXIT_CODE."
    exit $SCRIPT_EXIT_CODE
fi
echo "Bước huấn luyện và lưu trữ pipeline hoàn thành."

echo "\n===== PIPELINE HUẤN LUYỆN MÔ HÌNH TRÊN SERVER UBUNTU HOÀN TẤT ====="
echo "Pipeline và Label Encoder đã được lưu vào: $PROJECT_ROOT_DIR/app/model/latest_training_artifacts/"
echo "Vui lòng kiểm tra log và MLflow UI (http://192.168.28.98:5000) để xem chi tiết."