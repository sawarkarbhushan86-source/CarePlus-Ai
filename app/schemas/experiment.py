"""
CARE-PULSE AI — Experiment Schemas
Pydantic schemas for model performance benchmarks, evaluation metrics, and experiment tracking.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class ModelMetrics(BaseModel):
    model_name: str
    roc_auc: float
    pr_auc: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    specificity: float
    brier_score: float
    brier_skill_score: float
    expected_calibration_error: float
    confusion_matrix: Dict[str, int]  # tp, fp, tn, fn
    optimal_threshold: float


class ModelBenchmarkSummary(BaseModel):
    benchmark_timestamp: str
    models_evaluated: List[str]
    best_model_name: str
    best_roc_auc: float
    best_pr_auc: float
    metrics: Dict[str, ModelMetrics]
    dataset_fingerprint: str
    train_size: int
    test_size: int
    positive_prevalence_test: float


class ExperimentRunRecord(BaseModel):
    run_id: str
    model_name: str
    created_at: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]
    artifact_path: str
    notes: Optional[str] = None
