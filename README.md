# CARE-PULSE AI 🩺
### Explainable Healthcare Intelligence, Patient Readmission Risk Prediction & Hospital Analytics Platform

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.108+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Calibrated-FF6F00.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Interpretability-SHAP-success.svg)](https://shap.readthedocs.io/)
[![Tests](https://img.shields.io/badge/Pytest-25%2F25%20Passed-brightgreen.svg)]()

---

## 1. Executive Summary

**CARE-PULSE AI** is an end-to-end data science and clinical intelligence platform developed to analyze inpatient electronic health records (EHR), predict individual 30-day readmission risk, interpret predictive decision boundaries, and support operational hospital bed capacity planning.

Trained on the **UCI Diabetes 130-US Hospitals (1999–2008)** cohort ($N=101,766$ encounters across 71,518 unique patients), the platform tackles real-world healthcare analytics challenges:
- **Patient Leakage Prevention:** Eliminates data contamination across repeat patient visits via strict **patient-grouped cross-validation** (`patient_nbr`).
- **Probabilistic Calibration:** Employs Platt scaling (`CalibratedClassifierCV`) to slash Expected Calibration Error (ECE) by **86%** (down to 1.5%), ensuring predicted risks match empirical clinical probabilities.
- **Explainable Care Transitions:** Computes individual risk drivers via **TreeSHAP** to formulate personalized discharge navigator care plans.
- **Operations & Financial ROI Modeling:** Simulates bed-day throughput and models financial savings under CMS Hospital Readmissions Reduction Program (HRRP) benchmarks.

> **Clinical Disclaimer:** CARE-PULSE AI is an academic research and portfolio data science system. It is not an FDA-cleared diagnostic device and must not be used for autonomous clinical decisions or discharge denial.

---

## 2. Model Performance Benchmarks

All models were evaluated on an unseen, **patient-grouped test cohort** ($N=3,955$ encounters, baseline readmission prevalence = 11.25%):

| Model Architecture | ROC-AUC | PR-AUC (Avg Prec) | Sensitivity / Recall (@0.15) | Specificity (@0.15) | Brier Score | Expected Calibration Error (ECE) |
|---|---|---|---|---|---|---|
| **Logistic Regression (Baseline)** | 0.6626 | 0.2105 | 0.9978 | 0.0520 | 0.2286 | 0.3549 |
| **Random Forest (Bagging)** | 0.6645 | 0.2070 | 1.0000 | 0.0380 | 0.1806 | 0.2868 |
| **XGBoost (Gradient Boosted)** | 0.6565 | 0.1989 | 0.8742 | 0.4352 | 0.1120 | 0.1116 |
| **Calibrated XGBoost (Champion)** | **0.6658** | **0.2097** | **0.3281** | **0.8581** | **0.0962** | **0.0156** (-86% Error) |

### Key Clinical Risk Exposures (Odds Ratios with 95% Confidence Intervals)
- **Prior Inpatient Admissions ($\ge 1$):** $\text{OR} = \mathbf{2.27}\;[2.12, 2.44],\; p < 0.001$
- **Prior Emergency Department Visits ($\ge 1$):** $\text{OR} = \mathbf{1.88}\;[1.64, 2.15],\; p < 0.001$
- **Multimorbidity ($>7$ Diagnoses):** $\text{OR} = \mathbf{1.40}\;[1.30, 1.50],\; p < 0.001$
- **Polypharmacy ($>15$ Medications):** $\text{OR} = \mathbf{1.38}\;[1.28, 1.48],\; p < 0.001$
- **Prolonged Inpatient Stay ($>7$ Days):** $\text{OR} = \mathbf{1.35}\;[1.24, 1.48],\; p < 0.001$
- **Insulin Regimen Prescribed:** $\text{OR} = \mathbf{1.27}\;[1.18, 1.36],\; p < 0.001$
- **Untested Inpatient HbA1c:** $\text{OR} = \mathbf{1.18}\;[1.07, 1.30],\; p < 0.001$

---

## 3. Project Architecture

```
d:/CARE-PULSE AI/
├── app/
│   ├── main.py                  # FastAPI application entrypoint & lifespan
│   ├── config.py                # Pydantic Settings, clinical thresholds, directory paths
│   ├── api/                     # REST API v1 routes
│   │   ├── health.py            # Health & hardware diagnostics
│   │   ├── dataset_routes.py    # Schema verification, column profiling, sample data
│   │   ├── analytics_routes.py  # EDA statistics, Odds Ratios, Chi-square tests
│   │   ├── model_routes.py      # Benchmarks, ROC/PR curves, predictions
│   │   ├── explanation_routes.py# Global SHAP importance & local patient waterfalls
│   │   └── report_routes.py     # Executive reports & ROI simulation
│   ├── schemas/                 # Strongly-typed Pydantic request/response models
│   ├── services/                # Pure Python domain business logic services
│   │   ├── data_loader.py       # Fingerprinted dataset loading & caching
│   │   ├── data_validation.py   # Schema audits & clinical exclusions (expired/hospice)
│   │   ├── preprocessing.py     # ICD-9 grouping & patient-grouped split
│   │   ├── eda_service.py       # Cohort prevalence & utilization statistics
│   │   ├── statistics_service.py# Odds Ratios (95% CI) & Chi-Square tests
│   │   ├── training_service.py  # End-to-end model training & artifact saving
│   │   ├── evaluation_service.py# ROC-AUC, PR-AUC, ECE, Brier score
│   │   ├── calibration_service.py # Reliability diagrams & Platt scaling
│   │   ├── explainability_service.py # TreeSHAP & local risk attributions
│   │   ├── subgroup_analysis.py # Algorithmic fairness & demographic parity
│   │   ├── uncertainty_service.py# 95% Confidence Intervals for probabilities
│   │   ├── prediction_service.py# Real-time single & batch encounter inference
│   │   ├── operations_service.py# Hospital bed planning & nurse intervention ROI
│   │   └── reporting_service.py # Structured clinical audits & Markdown export
│   └── persistence/
│       └── experiment_repository.py # SQLite experiment run tracking
├── ui/
│   ├── dashboard.py             # Streamlit application entrypoint (Dark Theme)
│   ├── api_client.py            # Resilient API client with in-process fallback
│   └── pages/                   # 8 interactive dashboard pages
├── scripts/
│   ├── download_dataset.py      # Real UCI dataset acquisition script (OpenML 4541)
│   ├── validate_dataset.py      # Dataset validation CLI
│   ├── run_eda.py               # Exploratory analysis & Odds Ratio CLI
│   ├── train_models.py          # Model training CLI
│   └── run_benchmarks.py        # Benchmark comparison CLI
├── tests/                       # 25 automated pytest tests (100% pass)
├── data/                        # raw/, sample/, processed/
├── artifacts/                   # models/ (.joblib), metrics/ (.json), experiments.db
├── docs/                        # architecture, methodology, data_dictionary, model_card, limitations, deployment
├── requirements.txt             # Verified Python dependencies
├── Dockerfile                   # Multi-service container specification
└── docker-compose.yml           # Local multi-container orchestration
```

---

## 4. Quickstart Guide

### 4.1 Local Setup
```bash
# 1. Clone repository
git clone https://github.com/your-username/care-pulse-ai.git
cd care-pulse-ai

# 2. Set up virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # (On Windows)
source .venv/bin/activate       # (On Linux / Mac)

# 3. Install dependencies
pip install -r requirements.txt
```

### 4.2 Data Pipeline & Model Training
```bash
# Acquire authentic UCI dataset (101,766 records)
python scripts/download_dataset.py

# Validate schema & clinical exclusions
python scripts/validate_dataset.py

# Run EDA & Odds Ratios
python scripts/run_eda.py

# Train models & calibrate probabilities
python scripts/train_models.py --subsample 25000

# Run benchmarks & fairness audit
python scripts/run_benchmarks.py
```

### 4.3 Running Automated Tests
```bash
pytest tests/ -v
# Output: 25 passed in 13.08s
```

### 4.4 Launching the Application
In Terminal 1 (FastAPI Server):
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
# Access docs: http://localhost:8000/docs
```

In Terminal 2 (Streamlit Dashboard):
```bash
streamlit run ui/dashboard.py --server.port 8501
# Access UI: http://localhost:8501
```

---

## 5. Docker Deployment

Run both the FastAPI backend and Streamlit dashboard with a single command:
```bash
docker-compose up --build -d
```
- API Docs: `http://localhost:8000/docs`
- Streamlit UI: `http://localhost:8501`

---

## 6. Citation & Provenance
- **Dataset:** Strack, B. et al. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records.* BioMed Research International, vol. 2014. [UCI ML Repository ID: 296 / OpenML ID: 4541].
- **SHA-256 Fingerprint:** `e965ae6f7b59d3e9e47124135e3153efba436d644331f805e812d26388c1256b`
