"""
CARE-PULSE AI — Dataset Validation CLI Script
Validates raw and sample dataset schemas, audits missingness, detects repeat patients,
and checks clinical exclusion criteria.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.data_loader import data_loader_service
from app.services.data_validation import data_validation_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Initiating dataset validation audit...")
    try:
        df = data_loader_service.load_dataset(preferred_source="raw")
    except FileNotFoundError:
        logger.warning("Raw dataset not found. Falling back to sample dataset...")
        try:
            df = data_loader_service.load_dataset(preferred_source="sample")
        except FileNotFoundError as e:
            logger.error(f"Validation failed: {e}")
            sys.exit(1)

    report = data_validation_service.run_full_validation(df)

    print("\n" + "=" * 60)
    print("CARE-PULSE AI — DATASET QUALITY & INTEGRITY REPORT")
    print("=" * 60)
    print(f"Total Encounter Records:          {report.total_rows:,}")
    print(f"Total Clinical Features:          {report.total_columns}")
    print(f"Schema Validation Passed:         {'YES' if report.is_valid else 'NO'}")
    print(f"Unique Patient Identifiers:       {report.unique_patients_count:,}")
    print(f"Repeated Patient Encounters:      {report.repeated_patient_encounters_count:,}")
    print(f"30-Day Readmission Prevalence:    {report.target_rate_30d * 100:.2f}%")
    print(f"Clinical Exclusions (Deceased):   {report.excluded_encounters_count:,}")
    print(f"High-Missing Columns (>40%):      {', '.join(report.high_missingness_columns) or 'None'}")
    print("\nValidation Audit Log:")
    for msg in report.validation_messages:
        print(f"  [i] {msg}")
    print("=" * 60 + "\n")

    if not report.is_valid:
        logger.error("Dataset validation detected missing required schema features.")
        sys.exit(1)

    logger.info("Dataset validation passed successfully.")


if __name__ == "__main__":
    main()
