# -*- coding: utf-8 -*-
# app/preprocessing_module.py
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# Hàm tải tài nguyên NLTK (quan trọng khi triển khai)
def download_nltk_resources_for_api():
    # Trong môi trường production, bạn có thể muốn tải sẵn vào Docker image
    # hoặc có một vị trí cố định cho nltk_data
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('corpora/omw-1.4')
        nltk.data.find('tokenizers/punkt') # Cần cho word_tokenize nếu dùng, hoặc split() đơn giản
        print("Tài nguyên NLTK đã có sẵn.")
    except LookupError:
        print("Đang tải các tài nguyên NLTK cần thiết cho API...")
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True) # Phụ thuộc của wordnet
        nltk.download('punkt', quiet=True)
        print("Tải tài nguyên NLTK hoàn tất.")

# Gọi hàm tải khi module được import lần đầu
download_nltk_resources_for_api()

# Khởi tạo một lần để tăng tốc độ
stop_words_english = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text_for_prediction_api(text: str) -> str:
    """
    Hàm tiền xử lý văn bản cho API.
    Phải tương đồng với hàm tiền xử lý đã dùng khi huấn luyện.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text) # Loại bỏ số
    text = re.sub(r'[^\w\s]', '', text) # Loại bỏ dấu câu
    text = re.sub(r'\s+', ' ', text).strip() # Loại bỏ khoảng trắng thừa

    # Tokenization đơn giản bằng split() hoặc nltk.word_tokenize()
    # Nếu dùng nltk.word_tokenize(), đảm bảo 'punkt' đã được tải
    tokens = text.split() # Hoặc: tokens = nltk.word_tokenize(text)

    # Lemmatization và loại bỏ stop words
    processed_tokens = [
        lemmatizer.lemmatize(word, pos='v') for word in tokens
        if word not in stop_words_english and len(word) > 2 # Bỏ từ ngắn
    ]
    return " ".join(processed_tokens)

if __name__ == '__main__':
    # Test
    sample = "This is a test 123 for the API preprocessor!"
    print(f"Original: {sample}")
    print(f"Processed: {preprocess_text_for_prediction_api(sample)}")