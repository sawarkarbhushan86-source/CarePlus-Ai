# CARE-PULSE AI — Local & Production Deployment Guide

## 1. Quick Local Setup

### Prerequisites
- Python 3.11 (or compatible 3.10+)
- Git
- 4GB+ RAM

### Installation Steps
```bash
# 1. Clone repository
git clone https://github.com/your-username/care-pulse-ai.git
cd care-pulse-ai

# 2. Create and activate virtual environment
python -m venv .venv

# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment configuration
cp .env.example .env
```

---

## 2. Pipeline Execution Order

Execute the pipeline scripts sequentially from the repository root:

```bash
# 1. Acquire dataset (OpenML ID: 4541 mirror of UCI Diabetes 130-US Hospitals)
python scripts/download_dataset.py

# 2. Validate schema and check clinical exclusions
python scripts/validate_dataset.py

# 3. Run exploratory analysis & epidemiological odds ratios
python scripts/run_eda.py

# 4. Train model architectures and calibrate probabilities
python scripts/train_models.py --subsample 25000

# 5. Run benchmark audits and subgroup fairness analysis
python scripts/run_benchmarks.py
```

---

## 3. Starting the Applications

### 3.1 Start FastAPI Backend Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive OpenAPI Docs: `http://localhost:8000/docs`
- Redoc API Reference: `http://localhost:8000/redoc`
- Health Check: `http://localhost:8000/api/v1/health`

### 3.2 Start Streamlit Intelligence Dashboard
In a separate terminal:
```bash
streamlit run ui/dashboard.py --server.port 8501
```
- Dashboard Access: `http://localhost:8501`

---

## 4. Running Automated Tests

Run the full pytest suite:
```bash
pytest tests/ -v
```

---

## 5. Docker & Containerized Deployment

Run the complete multi-service application stack using Docker Compose:

```bash
# Build and run containers
docker-compose up --build -d

# View service logs
docker-compose logs -f

# Stop containers
docker-compose down
```
- Backend API will be mapped to `http://localhost:8000`
- Streamlit UI will be mapped to `http://localhost:8501`
