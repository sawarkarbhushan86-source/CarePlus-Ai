"""
CARE-PULSE AI — Dataset Schemas
Pydantic schemas for dataset validation, metadata, and quality summaries.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class DatasetMetadata(BaseModel):
    dataset_name: str
    source: str
    citation: str
    total_records: int
    total_columns: int
    columns: List[str]
    sha256: str
    target_distribution_raw: Dict[str, int]
    historical_cohort: str
    disclaimer: str


class ColumnQualitySummary(BaseModel):
    name: str
    dtype: str
    non_null_count: int
    null_count: int
    missing_percentage: float
    unique_values: int
    sample_values: List[Any]


class DataValidationReport(BaseModel):
    total_rows: int
    total_columns: int
    is_valid: bool
    missingness_summary: Dict[str, float]
    high_missingness_columns: List[str]
    duplicate_rows_count: int
    unique_patients_count: int
    repeated_patient_encounters_count: int
    target_distribution: Dict[str, int]
    target_rate_30d: float
    excluded_encounters_count: int
    validation_messages: List[str]


class EDAStatsSummary(BaseModel):
    total_encounters: int
    readmission_30d_prevalence: float
    avg_length_of_stay: float
    median_length_of_stay: float
    avg_num_medications: float
    avg_num_lab_procedures: float
    avg_num_procedures: float
    readmission_by_age: Dict[str, float]
    readmission_by_race: Dict[str, float]
    readmission_by_prior_inpatient: Dict[str, float]
    readmission_by_a1c: Dict[str, float]
    readmission_by_insulin: Dict[str, float]
