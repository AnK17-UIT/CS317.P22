#!/bin/bash

echo "===== BẮT ĐẦU PIPELINE HUẤN LUYỆN MÔ HÌNH TRÊN SERVER UBUNTU ====="
set -e

TRAINING_PIPELINE_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT_DIR=$(realpath "$TRAINING_PIPELINE_DIR/..") # Đây sẽ là /home/mlops/Lab-2-MLOPS/app/train_again

echo "Thư mục gốc dự án: $PROJECT_ROOT_DIR"
echo "Thư mục pipeline huấn luyện: $TRAINING_PIPELINE_DIR"

# --- KÍCH HOẠT MÔI TRƯỜNG ẢO ---
VENV_PATH="$PROJECT_ROOT_DIR/venv" # Đường dẫn chính xác đến thư mục venv

if [ -f "$VENV_PATH/bin/activate" ]; then
  echo "Kích hoạt môi trường ảo tại $VENV_PATH..."
  source "$VENV_PATH/bin/activate"
  echo "PYTHONPATH hiện tại (sau khi kích hoạt venv): '$PYTHONPATH'" # Kiểm tra PYTHONPATH
  echo "Đường dẫn thực thi Python (which python):"
  which python
  echo "Phiên bản Python (python --version):"
  python --version
else
  echo "CẢNH BÁO: Không tìm thấy file activate của môi trường ảo tại '$VENV_PATH/bin/activate'."
  echo "Vui lòng tạo và cài đặt các gói vào môi trường ảo trước."
  exit 1 # Thoát nếu không có venv
fi
# --- KẾT THÚC KÍCH HOẠT MÔI TRƯỜNG ẢO ---

# Tạo thư mục output nếu chưa có
mkdir -p "$PROJECT_ROOT_DIR/app/model/latest_training_artifacts" # Đường dẫn này có vẻ lạ, xem lại bên dưới
echo "Đảm bảo thư mục output cho artifacts tồn tại: $PROJECT_ROOT_DIR/app/model/latest_training_artifacts"

# Đường dẫn đến script huấn luyện chính
TRAIN_SCRIPT="$TRAINING_PIPELINE_DIR/train_model_pipeline.py"

echo "\n>>> Chạy Bước Chính: Huấn luyện và Lưu trữ Pipeline..."
# Gọi Python một cách tường minh, nó sẽ sử dụng Python từ venv đã kích hoạt
python "$TRAIN_SCRIPT"
if [ $? -ne 0 ]; then
    echo "LỖI: Bước huấn luyện và lưu trữ pipeline thất bại."
    exit 1
fi
echo "Bước huấn luyện và lưu trữ pipeline hoàn thành."

echo "\n===== PIPELINE HUẤN LUYỆN MÔ HÌNH TRÊN SERVER UBUNTU HOÀN TẤT ====="
# ... (phần còn lại) ...

# Deactivate venv (tùy chọn)
# echo "Deactivating virtual environment..."
# deactivate