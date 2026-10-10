"""
CARE-PULSE AI — Model Training CLI Script
Trains Logistic Regression, Random Forest, XGBoost, and Calibrated XGBoost
using patient-grouped cross-validation, saves artifacts, and records benchmarks.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.services.training_service import training_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train CARE-PULSE AI Readmission Risk Models")
    parser.add_argument(
        "--subsample",
        type=int,
        default=25000,
        help="Cohort subsample size for fast training (pass 0 for full 101,766 dataset)",
    )
    parser.add_argument(
        "--use-sample-data",
        action="store_true",
        help="Use the 5,000-encounter sample dataset instead of raw dataset",
    )
    args = parser.parse_args()

    subsample_size = None if args.subsample == 0 else args.subsample

    logger.info("Starting autonomous training pipeline...")
    summary = training_service.train_all_models(
        subsample_size=subsample_size,
        use_sample_data=args.use_sample_data,
    )

    print("\n" + "=" * 75)
    print("CARE-PULSE AI — MODEL BENCHMARK COMPARISON REPORT")
    print("=" * 75)
    print(f"Training Encounters:   {summary.train_size:,}")
    print(f"Test Encounters:       {summary.test_size:,} (Patient-Grouped Unseen Cohort)")
    print(f"Test Base Rate:        {summary.positive_prevalence_test * 100:.2f}% (30-day Readmitted)")
    print(f"Champion Architecture: {summary.best_model_name}")
    print("-" * 75)
    print(f"{'Model Architecture':<22} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'Sens/Recall':<11} | {'Brier':<8} | {'ECE':<8}")
    print("-" * 75)
    for model_name, m in summary.metrics.items():
        print(
            f"{model_name:<22} | "
            f"{m.roc_auc:<8.4f} | "
            f"{m.pr_auc:<8.4f} | "
            f"{m.recall:<11.4f} | "
            f"{m.brier_score:<8.4f} | "
            f"{m.expected_calibration_error:<8.4f}"
        )
    print("=" * 75)
    print(f"\nArtifacts successfully written to: {settings.MODELS_DIR}")
    print(f"Benchmark summary written to:   {settings.METRICS_DIR / 'benchmark_summary.json'}\n")


if __name__ == "__main__":
    main()
