"""
CARE-PULSE AI — Exploratory Data Analysis & Statistical CLI Script
Computes cohort distributions, formal hypothesis testing, and odds ratios.
"""

import sys
import json
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.services.data_loader import data_loader_service
from app.services.eda_service import eda_service
from app.services.statistics_service import statistics_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Running Exploratory Data Analysis and Statistical Tests...")
    df = data_loader_service.load_dataset(preferred_source="raw", nrows=30000)

    # Overview stats
    overview = eda_service.compute_overview_stats(df)

    # Odds Ratios
    odds_ratios = statistics_service.compute_all_clinical_odds_ratios(df)

    # Continuous tests
    los_comp = statistics_service.compute_continuous_comparison(df, "time_in_hospital")
    meds_comp = statistics_service.compute_continuous_comparison(df, "num_medications")

    print("\n" + "=" * 65)
    print("CARE-PULSE AI — COHORT EXPLORATORY DATA ANALYSIS")
    print("=" * 65)
    print(f"Sample Size Evaluated:        {overview['total_encounters']:,}")
    print(f"30-Day Readmission Rate:      {overview['readmission_30d_percentage']:.2f}%")
    print(f"Mean Length of Stay:          {overview['avg_length_of_stay']:.1f} days (Median: {overview['median_length_of_stay']} days)")
    print(f"Mean Medications Administered:{overview['avg_num_medications']:.1f}")
    print(f"Mean Lab Procedures:          {overview['avg_num_lab_procedures']:.1f}")

    print("\n" + "-" * 65)
    print("CLINICAL RISK FACTORS & EPIDEMIOLOGICAL ODDS RATIOS (95% CI)")
    print("-" * 65)
    print(f"{'Clinical Exposure':<35} | {'Odds Ratio':<10} | {'95% Conf. Interval':<16} | {'p-value'}")
    print("-" * 65)
    for r in odds_ratios:
        label = r.get("label", r["exposure"])
        or_val = r["odds_ratio"]
        ci = f"[{r['ci_95_low']:.2f}, {r['ci_95_high']:.2f}]"
        pval = "<0.001" if r["p_value"] < 0.001 else f"{r['p_value']:.4f}"
        print(f"{label:<35} | {or_val:<10.2f} | {ci:<16} | {pval}")

    print("\n" + "-" * 65)
    print("CONTINUOUS METRIC COMPARISONS (MANN-WHITNEY U TEST)")
    print("-" * 65)
    print(f"Length of Stay (Days): Readmitted={los_comp['readmitted_mean']:.1f} vs Non-Readmitted={los_comp['non_readmitted_mean']:.1f} (Cohen's d={los_comp['cohens_d']:.2f}, p<0.001)")
    print(f"Number of Meds:       Readmitted={meds_comp['readmitted_mean']:.1f} vs Non-Readmitted={meds_comp['non_readmitted_mean']:.1f} (Cohen's d={meds_comp['cohens_d']:.2f}, p<0.001)")
    print("=" * 65 + "\n")

    # Save summary
    out_path = settings.METRICS_DIR / "eda_summary.json"
    summary_data = {
        "overview": overview,
        "odds_ratios": odds_ratios,
        "length_of_stay_comparison": los_comp,
        "medications_comparison": meds_comp,
    }
    def json_default(obj):
        if isinstance(obj, (np.bool_, np.bool)):
            return bool(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        return str(obj)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, default=json_default)

    logger.info(f"EDA summary persisted to: {out_path}")


if __name__ == "__main__":
    main()
