"""
CARE-PULSE AI — Configuration Module
Centralized settings, file paths, model hyperparameters, and operational parameters.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # Project Identity
    PROJECT_NAME: str = "CARE-PULSE AI"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # Directories
    BASE_DIR: Path = PROJECT_ROOT
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    SAMPLE_DATA_DIR: Path = DATA_DIR / "sample"
    ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
    MODELS_DIR: Path = ARTIFACTS_DIR / "models"
    METRICS_DIR: Path = ARTIFACTS_DIR / "metrics"
    REPORTS_DIR: Path = ARTIFACTS_DIR / "reports"
    DB_PATH: Path = ARTIFACTS_DIR / "experiments.db"

    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    UI_PORT: int = 8501

    # Healthcare Analytics Parameters
    TARGET_COLUMN_RAW: str = "readmitted"
    TARGET_COLUMN_BINARY: str = "readmission_30d"
    POSITIVE_CLASS_VALUE: str = "<30"
    PATIENT_ID_COLUMN: str = "patient_nbr"
    ENCOUNTER_ID_COLUMN: str = "encounter_id"

    # Default Clinical Decision Thresholds
    DEFAULT_RISK_THRESHOLD: float = 0.15  # Tuned for high-recall in ~11% prevalence setting
    LOW_RISK_CUTOFF: float = 0.10
    HIGH_RISK_CUTOFF: float = 0.25

    # Financial / Operational Defaults (Hypothetical Hospital Model)
    AVG_READMISSION_COST_USD: float = 14500.0  # Average US Medicare 30d readmission cost
    INTERVENTION_COST_PER_PATIENT_USD: float = 75.0  # Post-discharge nurse phone call & med review
    INTERVENTION_EFFECTIVENESS: float = 0.22  # Estimated 22% relative reduction in readmission

    # Model Configuration
    RANDOM_STATE: int = 42
    TEST_SPLIT_RATIO: float = 0.20

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()

# Ensure directories exist
for path in [
    settings.DATA_DIR,
    settings.RAW_DATA_DIR,
    settings.PROCESSED_DATA_DIR,
    settings.SAMPLE_DATA_DIR,
    settings.ARTIFACTS_DIR,
    settings.MODELS_DIR,
    settings.METRICS_DIR,
    settings.REPORTS_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)
