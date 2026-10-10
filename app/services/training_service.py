"""
CARE-PULSE AI — Model Training and Benchmarking Service
Trains and benchmarks clinically relevant classifiers:
1. Logistic Regression (Interpretable Clinical Baseline)
2. Random Forest (Bagging Ensemble)
3. XGBoost (Gradient Boosted Decision Trees)
4. Calibrated Ensembles (Platt Scaled)
Logs metrics to SQLite repository and serializes trusted artifacts with Joblib.
"""

import os
import json
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

from app.config import settings
from app.services.data_loader import data_loader_service
from app.services.preprocessing import preprocessing_service
from app.services.evaluation_service import evaluation_service
from app.services.calibration_service import calibration_service
from app.persistence.experiment_repository import experiment_repo
from app.schemas.experiment import ModelBenchmarkSummary, ModelMetrics

logger = logging.getLogger(__name__)


class TrainingService:
    """Orchestrates end-to-end model training, probability calibration, and benchmarking."""

    def __init__(self):
        self.trained_models: Dict[str, Any] = {}
        self.benchmark_summary: Optional[ModelBenchmarkSummary] = None

    def train_all_models(
        self,
        subsample_size: Optional[int] = 25000,
        random_state: int = 42,
        use_sample_data: bool = False,
    ) -> ModelBenchmarkSummary:
        """
        Train and evaluate all candidate model architectures:
        Logistic Regression, Random Forest, XGBoost, and Calibrated XGBoost.
        """
        logger.info(f"Starting model training pipeline (subsample_size={subsample_size})...")

        # 1. Load data
        preferred = "sample" if use_sample_data else "raw"
        df = data_loader_service.load_dataset(preferred_source=preferred)

        if subsample_size and len(df) > subsample_size:
            logger.info(f"Subsampling cohort to {subsample_size} encounters for training speed...")
            df = df.sample(n=subsample_size, random_state=random_state).reset_index(drop=True)

        # 2. Prepare patient-grouped data
        split_data = preprocessing_service.prepare_modeling_data(
            df, test_size=settings.TEST_SPLIT_RATIO, random_state=random_state
        )

        X_train = split_data["X_train"]
        y_train = split_data["y_train"]
        X_test = split_data["X_test"]
        y_test = split_data["y_test"]
        num_cols = split_data["num_cols"]
        cat_cols = split_data["cat_cols"]

        meta = data_loader_service.get_metadata()
        fingerprint = meta.get("sha256", "UNKNOWN")

        results: Dict[str, ModelMetrics] = {}
        models_to_train = self._get_model_definitions(random_state)

        for model_name, classifier in models_to_train.items():
            logger.info(f"Training architecture: {model_name}...")
            preprocessor = preprocessing_service.build_preprocessor_pipeline(num_cols, cat_cols)

            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", classifier),
                ]
            )

            # Fit pipeline
            pipeline.fit(X_train, y_train)

            # Predict probabilities on unseen patient-grouped test set
            y_prob = pipeline.predict_proba(X_test)[:, 1]

            # Evaluate metrics
            metrics = evaluation_service.evaluate_model(
                model_name=model_name,
                y_true=y_test.values,
                y_prob=y_prob,
                threshold=settings.DEFAULT_RISK_THRESHOLD,
            )
            results[model_name] = metrics

            # Save model artifact
            artifact_file = settings.MODELS_DIR / f"{model_name.lower().replace(' ', '_')}.joblib"
            joblib.dump(pipeline, artifact_file)
            self.trained_models[model_name] = pipeline

            # Log to SQLite repository
            run_id = f"{model_name.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"
            experiment_repo.log_run(
                run_id=run_id,
                model_name=model_name,
                hyperparameters={"type": model_name, "random_state": random_state},
                metrics=metrics.dict(),
                dataset_fingerprint=fingerprint,
                artifact_path=str(artifact_file),
                notes=f"Patient-grouped test evaluation (N_test={len(y_test)})",
            )
            logger.info(f"Completed {model_name}. ROC-AUC: {metrics.roc_auc:.4f}, PR-AUC: {metrics.pr_auc:.4f}")

        # Train Calibrated XGBoost
        logger.info("Training Calibrated XGBoost (Platt Scaling)...")
        xgb_pipeline = self.trained_models.get("XGBoost")
        if xgb_pipeline is not None:
            preprocessor = preprocessing_service.build_preprocessor_pipeline(num_cols, cat_cols)
            X_train_trans = preprocessor.fit_transform(X_train)
            X_test_trans = preprocessor.transform(X_test)

            base_xgb = models_to_train["XGBoost"]
            calibrated_clf = CalibratedClassifierCV(estimator=base_xgb, method="sigmoid", cv=3)
            calibrated_clf.fit(X_train_trans, y_train)

            calibrated_pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", calibrated_clf),
                ]
            )

            y_prob_cal = calibrated_pipeline.predict_proba(X_test)[:, 1]
            cal_metrics = evaluation_service.evaluate_model(
                model_name="Calibrated XGBoost",
                y_true=y_test.values,
                y_prob=y_prob_cal,
                threshold=settings.DEFAULT_RISK_THRESHOLD,
            )
            results["Calibrated XGBoost"] = cal_metrics

            cal_artifact = settings.MODELS_DIR / "calibrated_xgboost.joblib"
            joblib.dump(calibrated_pipeline, cal_artifact)
            self.trained_models["Calibrated XGBoost"] = calibrated_pipeline

            run_id = f"calibrated_xgboost_{uuid.uuid4().hex[:8]}"
            experiment_repo.log_run(
                run_id=run_id,
                model_name="Calibrated XGBoost",
                hyperparameters={"type": "CalibratedClassifierCV", "base": "XGBoost", "method": "sigmoid"},
                metrics=cal_metrics.dict(),
                dataset_fingerprint=fingerprint,
                artifact_path=str(cal_artifact),
                notes="Platt scaled 3-fold cross-validated XGBoost",
            )

        # Identify champion model by PR-AUC (or ROC-AUC)
        best_model_name = max(results.keys(), key=lambda m: results[m].roc_auc)

        # Save test evaluation dataset for explainability and evaluation cache
        eval_cache_path = settings.METRICS_DIR / "test_evaluation_cache.joblib"
        joblib.dump(
            {
                "X_test": X_test,
                "y_test": y_test,
                "raw_test_df": split_data["raw_test_df"],
            },
            eval_cache_path,
        )

        benchmark_summary = ModelBenchmarkSummary(
            benchmark_timestamp=datetime.utcnow().isoformat(),
            models_evaluated=list(results.keys()),
            best_model_name=best_model_name,
            best_roc_auc=results[best_model_name].roc_auc,
            best_pr_auc=results[best_model_name].pr_auc,
            metrics=results,
            dataset_fingerprint=fingerprint,
            train_size=len(X_train),
            test_size=len(X_test),
            positive_prevalence_test=round(float(y_test.mean()), 4),
        )

        # Serialize benchmark summary to JSON
        benchmark_json_path = settings.METRICS_DIR / "benchmark_summary.json"
        with open(benchmark_json_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_summary.dict(), f, indent=2)

        self.benchmark_summary = benchmark_summary
        logger.info(f"Benchmark summary saved to: {benchmark_json_path}. Champion: {best_model_name}")

        return benchmark_summary

    def load_model(self, model_name: str = "XGBoost") -> Any:
        """Load a saved model pipeline by name."""
        clean_name = model_name.lower().replace(" ", "_")
        artifact_path = settings.MODELS_DIR / f"{clean_name}.joblib"

        if not artifact_path.exists():
            # Check for champion or any available model
            available = list(settings.MODELS_DIR.glob("*.joblib"))
            if available:
                artifact_path = available[0]
            else:
                raise FileNotFoundError(f"Model artifact not found for '{model_name}'. Please trigger training first.")

        return joblib.load(artifact_path)

    def get_benchmark_summary(self) -> Optional[ModelBenchmarkSummary]:
        """Load or return cached benchmark summary."""
        if self.benchmark_summary is not None:
            return self.benchmark_summary

        json_path = settings.METRICS_DIR / "benchmark_summary.json"
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return ModelBenchmarkSummary(**data)
            except Exception as e:
                logger.error(f"Error reading benchmark summary JSON: {e}")
        return None

    def _get_model_definitions(self, random_state: int) -> Dict[str, Any]:
        """Define candidate classifiers with regularized clinical hyperparameters."""
        return {
            "Logistic Regression": LogisticRegression(
                penalty="l2",
                C=1.0,
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state,
                solver="lbfgs",
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=4,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=-1,
            ),
            "XGBoost": xgb.XGBClassifier(
                n_estimators=120,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=2.5,  # Moderate weighting for imbalanced positive rate (~11%)
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
            ),
        }


training_service = TrainingService()
