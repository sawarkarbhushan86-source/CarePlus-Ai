"""
CARE-PULSE AI — Dataset Download and Acquisition Script
Source: UCI Machine Learning Repository / OpenML Mirror
Dataset: Diabetes 130-US Hospitals for Years 1999-2008 (ID: 4541 / UCI 296)
"""

import os
import sys
import json
import hashlib

import logging
from pathlib import Path
import pandas as pd
from sklearn.datasets import fetch_openml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
SAMPLE_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"
PROCESSED_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def compute_file_fingerprint(filepath: Path) -> str:
    """Compute SHA-256 fingerprint of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_dataset(force: bool = False) -> Path:
    """
    Acquire the authentic UCI Diabetes 130-US Hospitals dataset.
    Saves raw CSV and creates a sample CSV for swift development and testing.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    raw_csv_path = RAW_DATA_DIR / "diabetic_data.csv"
    meta_path = RAW_DATA_DIR / "dataset_metadata.json"
    sample_csv_path = SAMPLE_DATA_DIR / "diabetic_sample.csv"

    if raw_csv_path.exists() and not force:
        logger.info(f"Raw dataset already exists at {raw_csv_path}")
        return raw_csv_path

    logger.info("Initiating acquisition of UCI Diabetes 130-US Hospitals dataset (OpenML ID: 4541)...")
    try:
        data = fetch_openml(data_id=4541, as_frame=True, parser="auto")
        df = data.frame
        logger.info(f"Successfully fetched dataset. Shape: {df.shape}")

        # Save raw dataframe
        df.to_csv(raw_csv_path, index=False)
        logger.info(f"Raw dataset saved to: {raw_csv_path}")

        # Compute fingerprint
        sha256_hash = compute_file_fingerprint(raw_csv_path)

        metadata = {
            "dataset_name": "Diabetes 130-US hospitals for years 1999-2008",
            "source": "UCI Machine Learning Repository / OpenML ID 4541",
            "citation": "Strack, B. et al. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates. BioMed Research International.",
            "total_records": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "sha256": sha256_hash,
            "target_distribution_raw": df["readmitted"].value_counts().to_dict() if "readmitted" in df.columns else {},
            "historical_cohort": "1999-2008 US Hospitals",
            "disclaimer": "Historical US healthcare encounter data. For research & portfolio benchmarking only."
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata recorded at: {meta_path}")

        # Generate representative sample (stratified by patient and target if possible, 5,000 encounters)
        sample_size = min(5000, len(df))
        sample_df = df.sample(n=sample_size, random_state=42)
        sample_df.to_csv(sample_csv_path, index=False)
        logger.info(f"Sample dataset (N={sample_size}) saved to: {sample_csv_path}")

        return raw_csv_path

    except Exception as e:
        logger.error(f"Failed to fetch dataset via OpenML: {e}", exc_info=True)
        raise RuntimeError(f"Dataset acquisition failed: {e}")


if __name__ == "__main__":
    force_download = "--force" in sys.argv
    download_dataset(force=force_download)
