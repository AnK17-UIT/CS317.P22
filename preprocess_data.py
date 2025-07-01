# training_pipeline/preprocess_data.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import sys

# --- Thiết lập đường dẫn để import module dùng chung ---
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_DIR)

# --- Imports ---
try:
    # Đổi tên import để rõ ràng hơn
    from app.preprocessing_module import preprocess_text, download_nltk_resources
    import config_training as config
except ImportError as e:
    print(f"FATAL: Không thể import các module cần thiết. Lỗi: {e}")
    print("Hãy chắc chắn rằng bạn đang chạy script từ thư mục gốc của dự án hoặc training_pipeline, và cấu trúc thư mục là chính xác.")
    raise

def load_and_preprocess_data():
    """
    Tải dữ liệu thô, kết hợp các cột text, áp dụng hàm tiền xử lý
    và mã hóa nhãn. Trả về None nếu có lỗi.
    """
    print("\n--- BƯỚC 1: BẮT ĐẦU TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU ---")
    
    try:
        # 1. Đảm bảo có tài nguyên NLTK
        download_nltk_resources()
        
        # 2. Tải dữ liệu
        df = pd.read_csv(config.RAW_DATA_PATH, on_bad_lines="skip", engine='python')
        print(f"Đã tải {len(df)} dòng từ {config.RAW_DATA_PATH}")

        # 3. Xử lý dữ liệu
        df.fillna('', inplace=True)
        required_columns = ['case_title', 'case_text', config.LABEL_COLUMN]
        if not all(col in df.columns for col in required_columns):
            print(f"LỖI: Thiếu một trong các cột cần thiết: {required_columns}")
            return None, None
            
        df['text_to_process'] = df['case_title'] + " " + df['case_text']
        
        print(f"Đang tiền xử lý văn bản bằng phương pháp: '{config.TEXT_PROCESSING_TYPE}'...")
        df['processed_text'] = df['text_to_process'].apply(
            lambda text: preprocess_text(text, processing_type=config.TEXT_PROCESSING_TYPE)
        )
        
        df.dropna(subset=['processed_text'], inplace=True)
        df = df[df['processed_text'].str.strip().astype(bool)]
        if df.empty:
            print("LỖI: Không còn dữ liệu sau khi tiền xử lý.")
            return None, None
        print(f"Còn lại {len(df)} dòng hợp lệ sau khi tiền xử lý.")

        # 4. Mã hóa nhãn
        le = LabelEncoder()
        df['case_outcome_num'] = le.fit_transform(df[config.LABEL_COLUMN])
        num_classes = len(le.classes_)
        print(f"Đã mã hóa nhãn thành {num_classes} lớp.")
        
        config.XGB_PARAMS['num_class'] = num_classes

        print("--- BƯỚC 1: HOÀN TẤT ---")
        return df, le

    except FileNotFoundError:
        print(f"LỖI NGHIÊM TRỌNG: Không tìm thấy file dữ liệu tại '{config.RAW_DATA_PATH}'.")
        return None, None
    except Exception as e:
        print(f"LỖI NGHIÊM TRỌNG trong quá trình tiền xử lý: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    df_processed, label_encoder = load_and_preprocess_data()
    if df_processed is not None:
        print("\n--- KIỂM TRA KẾT QUẢ TIỀN XỬ LÝ ---")
        print(df_processed[['processed_text', 'case_outcome_num']].head())
        print(f"\nCác lớp đã mã hóa: {list(label_encoder.classes_)}")