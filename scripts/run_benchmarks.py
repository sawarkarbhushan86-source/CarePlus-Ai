"""
CARE-PULSE AI — Model Benchmark & Fairness Evaluation CLI Script
Evaluates discrimination, probability calibration, threshold sensitivity, and subgroup fairness.
"""

import sys
import logging
from pathlib import Path
import joblib

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.services.training_service import training_service
from app.services.evaluation_service import evaluation_service
from app.services.subgroup_analysis import subgroup_analysis_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Running comprehensive model benchmarks and subgroup fairness audits...")
    eval_cache_file = settings.METRICS_DIR / "test_evaluation_cache.joblib"

    if not eval_cache_file.exists():
        logger.warning("Evaluation cache not found. Triggering model training first...")
        training_service.train_all_models(subsample_size=15000)

    cache_data = joblib.load(eval_cache_file)
    X_test = cache_data["X_test"]
    y_test = cache_data["y_test"].values
    raw_test_df = cache_data.get("raw_test_df", X_test)

    summary = training_service.get_benchmark_summary()
    if summary:
        print("\n" + "=" * 70)
        print("OVERALL MODEL PERFORMANCE BENCHMARKS (PATIENT-GROUPED TEST SET)")
        print("=" * 70)
        for name, m in summary.metrics.items():
            print(f"Model: {name:<22} | ROC-AUC: {m.roc_auc:.4f} | PR-AUC: {m.pr_auc:.4f} | ECE: {m.expected_calibration_error:.4f}")

    # Threshold sensitivity on champion model
    champion_name = summary.best_model_name if summary else "Calibrated XGBoost"
    pipeline = training_service.load_model(champion_name)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print("\n" + "-" * 70)
    print(f"THRESHOLD SENSITIVITY ANALYSIS ({champion_name})")
    print("-" * 70)
    print(f"{'Threshold':<10} | {'Sensitivity':<12} | {'Specificity':<12} | {'Precision':<10} | {'F1-Score'}")
    print("-" * 70)
    for t in [0.08, 0.10, 0.12, 0.15, 0.20, 0.25]:
        m = evaluation_service.evaluate_model(champion_name, y_test, y_prob, threshold=t)
        print(f"{t:<10.2f} | {m.recall:<12.4f} | {m.specificity:<12.4f} | {m.precision:<10.4f} | {m.f1_score:.4f}")

    # Subgroup Fairness
    print("\n" + "-" * 70)
    print(f"SUBGROUP FAIRNESS & CALIBRATION AUDIT ({champion_name}, Threshold=0.15)")
    print("-" * 70)
    for attr in ["gender", "race"]:
        if attr in raw_test_df.columns:
            subgroups = subgroup_analysis_service.evaluate_subgroups(raw_test_df, y_test, y_prob, attr, threshold=0.15)
            print(f"\n[Subgroup Demographic: {attr.upper()}]")
            print(f"{'Category':<20} | {'Sample N':<8} | {'ROC-AUC':<8} | {'Recall':<8} | {'FPR':<8} | {'Disp vs Avg'}")
            for s in subgroups:
                print(f"{s['subgroup']:<20} | {s['sample_size']:<8} | {s['roc_auc']:<8.4f} | {s['sensitivity_recall']:<8.4f} | {s['false_positive_rate']:<8.4f} | {s['tpr_disparity']:+.4f}")

    print("=" * 70 + "\n")
    logger.info("Benchmark evaluation completed.")


if __name__ == "__main__":
    main()
