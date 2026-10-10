"""
CARE-PULSE AI — Data Loader Service
Safe, validated, and cached loading of raw, processed, and sample healthcare encounter datasets.
"""

import json
import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import pandas as pd
from app.config import settings

logger = logging.getLogger(__name__)


class DataLoaderService:
    """Manages loading of raw and processed hospital encounter datasets with fingerprinting."""

    def __init__(self):
        self._cached_df: Optional[pd.DataFrame] = None
        self._cached_source: Optional[str] = None

    def get_raw_filepath(self) -> Path:
        return settings.RAW_DATA_DIR / "diabetic_data.csv"

    def get_sample_filepath(self) -> Path:
        return settings.SAMPLE_DATA_DIR / "diabetic_sample.csv"

    def get_processed_filepath(self) -> Path:
        return settings.PROCESSED_DATA_DIR / "cleaned_encounters.csv"

    def get_metadata_filepath(self) -> Path:
        return settings.RAW_DATA_DIR / "dataset_metadata.json"

    def compute_sha256(self, filepath: Path) -> str:
        """Calculate SHA-256 fingerprint for data integrity verification."""
        if not filepath.exists():
            return "FILE_NOT_FOUND"
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(8192 * 1024):
                sha256.update(chunk)
        return sha256.hexdigest()

    def get_metadata(self) -> Dict[str, Any]:
        """Read recorded metadata JSON or generate fallback metadata."""
        meta_file = self.get_metadata_filepath()
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error reading metadata file: {e}")

        raw_path = self.get_raw_filepath()
        return {
            "dataset_name": "Diabetes 130-US hospitals for years 1999-2008",
            "source": "UCI Machine Learning Repository (OpenML ID: 4541)",
            "citation": "Strack et al. (2014) BioMed Research International",
            "sha256": self.compute_sha256(raw_path) if raw_path.exists() else "NOT_DOWNLOADED",
            "historical_cohort": "1999-2008 US Hospitals",
            "disclaimer": "Historical US healthcare encounter data. For research & portfolio benchmarking only."
        }

    def load_dataset(self, preferred_source: str = "raw", nrows: Optional[int] = None, force_reload: bool = False) -> pd.DataFrame:
        """
        Load dataset. Prefers 'processed', 'raw', or 'sample'.
        Fallback to sample if raw is missing.
        """
        if not force_reload and self._cached_df is not None and self._cached_source == preferred_source and nrows is None:
            return self._cached_df

        target_path: Optional[Path] = None
        if preferred_source == "processed":
            p = self.get_processed_filepath()
            if p.exists():
                target_path = p
        elif preferred_source == "raw":
            p = self.get_raw_filepath()
            if p.exists():
                target_path = p
        elif preferred_source == "sample":
            p = self.get_sample_filepath()
            if p.exists():
                target_path = p

        # Fallback hierarchy
        if target_path is None or not target_path.exists():
            if self.get_raw_filepath().exists():
                target_path = self.get_raw_filepath()
            elif self.get_sample_filepath().exists():
                target_path = self.get_sample_filepath()
            else:
                raise FileNotFoundError(
                    f"Healthcare encounter dataset not found at {self.get_raw_filepath()}. "
                    "Please run 'python scripts/download_dataset.py' to acquire the UCI dataset."
                )

        logger.info(f"Loading healthcare encounter dataset from: {target_path} (nrows={nrows})")
        df = pd.read_csv(target_path, nrows=nrows, low_memory=False)

        # Standardize missing value symbols
        df = df.replace("?", None)

        if nrows is None:
            self._cached_df = df
            self._cached_source = preferred_source

        return df

    def get_dataset_summary(self) -> Dict[str, Any]:
        """Produce an immediate high-level overview of the dataset status and dimensions."""
        raw_exists = self.get_raw_filepath().exists()
        sample_exists = self.get_sample_filepath().exists()
        processed_exists = self.get_processed_filepath().exists()

        if not (raw_exists or sample_exists):
            return {
                "status": "dataset_missing",
                "message": "Dataset not found. Run scripts/download_dataset.py to acquire.",
                "raw_exists": False,
                "sample_exists": False,
                "processed_exists": False,
            }

        df = self.load_dataset(preferred_source="raw" if raw_exists else "sample", nrows=1000)
        meta = self.get_metadata()

        return {
            "status": "available",
            "raw_exists": raw_exists,
            "sample_exists": sample_exists,
            "processed_exists": processed_exists,
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "preview_sample_size": len(df),
            "metadata": meta,
            "fingerprint": meta.get("sha256", "UNKNOWN")
        }


data_loader_service = DataLoaderService()
