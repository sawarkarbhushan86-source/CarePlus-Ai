"""
Unit tests for CARE-PULSE AI Evaluation Service.
"""

import pytest
import numpy as np
from app.services.evaluation_service import evaluation_service
from app.schemas.experiment import ModelMetrics


def test_evaluation_metrics_computation():
    y_true = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.05, 0.10, 0.12, 0.20, 0.25, 0.30, 0.40, 0.65, 0.80, 0.90])

    metrics = evaluation_service.evaluate_model(
        model_name="TestModel", y_true=y_true, y_prob=y_prob, threshold=0.35
    )

    assert isinstance(metrics, ModelMetrics)
    assert metrics.model_name == "TestModel"
    assert 0.0 <= metrics.roc_auc <= 1.0
    assert 0.0 <= metrics.pr_auc <= 1.0
    assert 0.0 <= metrics.brier_score <= 1.0
    assert 0.0 <= metrics.expected_calibration_error <= 1.0

    cm = metrics.confusion_matrix
    assert (cm["tp"] + cm["fp"] + cm["tn"] + cm["fn"]) == 10
    assert cm["tp"] == 4
    assert cm["fp"] == 0


def test_expected_calibration_error():
    # Perfect calibration test
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.0, 0.0, 1.0, 1.0])

    ece = evaluation_service.compute_expected_calibration_error(y_true, y_prob, n_bins=5)
    assert ece == 0.0 or ece < 1e-4
