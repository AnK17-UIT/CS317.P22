# Legal Case Classification Project (MLOps)

This project builds an end-to-end MLOps pipeline using XGBoost to classify legal case outcomes from text data. It includes data processing, model training, experiment tracking, deployment via FastAPI, and monitoring with Prometheus & Grafana.

## 1. Setup Instructions

### Prerequisites

- Python >= 3.9
- Docker & Docker Compose
- Git

### Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
```

### Install Python Dependencies

Install packages using pinned versions:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```
pandas==2.0.3
scikit-learn==1.3.2
xgboost==2.0.3
nltk==3.8.1
joblib==1.3.2
fastapi==0.104.1
uvicorn[standard]==0.24.0.post1
mlflow==2.10.2
```

### NLTK Downloads

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')
```

### Prepare Dataset

Place your CSV data (e.g., `legal_text.csv`) in the `data/` folder.

Update paths in config or scripts accordingly.

## 2. Run Training Pipeline

```bash
python scripts/_01_preprocess_data.py
python scripts/_02_train_model.py
python scripts/_03_evaluate_model.py
```

Artifacts are saved in `trained_models/`. MLflow is used for experiment tracking.

Start MLflow server via Docker:

```bash
docker-compose up -d mlflow
```

Access MLflow UI at [http://localhost:5000](http://localhost:5000).

## 3. Serve Model with FastAPI

Ensure trained pipeline files are in `app/model/`.

Run the API:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Visit [http://localhost:8000/docs](http://localhost:8000/docs) to test the API.

## 4. Monitoring with Prometheus & Grafana

Start Prometheus and Grafana:

```bash
docker-compose up -d prometheus grafana
```

- Prometheus: [http://localhost:9090](http://localhost:9090)
- Grafana: [http://localhost:3000](http://localhost:3000) (login: admin/admin)

Metrics endpoint exposed via `/metrics` on FastAPI.

## 5. Folder Structure

```
project_root/
├── app/                  # FastAPI app
│   ├── main.py
│   └── model/
├── data/                 # Input data
├── trained_models/       # Trained model parts
├── scripts/              # Training scripts
├── prometheus/           # Prometheus config
├── docker-compose.yml
├── requirements.txt
└── README.md
```
