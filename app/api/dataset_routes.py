"""
CARE-PULSE AI — Dataset Management API Routes
"""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Query, HTTPException
from app.services.data_loader import data_loader_service
from app.services.data_validation import data_validation_service
from app.schemas.dataset import DataValidationReport, ColumnQualitySummary

router = APIRouter(prefix="/dataset", tags=["Dataset Management"])


@router.get("/summary", summary="Dataset Fingerprint and Cohort Dimensions")
def get_dataset_summary():
    """Retrieve SHA-256 fingerprint, source citations, and file availability."""
    return data_loader_service.get_dataset_summary()


@router.get("/validation", response_model=DataValidationReport, summary="Run Data Quality & Schema Audit")
def get_data_validation():
    """Run full schema check, missingness analysis, duplicate scan, and clinical exclusion audit."""
    import json
    from app.config import settings
    val_file = settings.METRICS_DIR / "full_validation_report.json"
    if val_file.exists():
        try:
            with open(val_file, "r") as f:
                return DataValidationReport(**json.load(f))
        except Exception:
            pass

    try:
        df = data_loader_service.load_dataset(preferred_source="raw", nrows=10000)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return data_validation_service.run_full_validation(df)


@router.get("/columns", response_model=List[ColumnQualitySummary], summary="Column-by-Column Data Profiling")
def get_column_summaries():
    """Inspect data types, non-null counts, missing rates, and sample values for each feature."""
    try:
        df = data_loader_service.load_dataset(preferred_source="raw", nrows=5000)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return data_validation_service.get_column_quality_summaries(df)


@router.get("/sample", summary="Retrieve Tabular Encounter Records")
def get_dataset_sample(n: int = Query(default=20, ge=1, le=200)):
    """Fetch sample rows for interactive data grid display."""
    try:
        df = data_loader_service.load_dataset(preferred_source="raw", nrows=n)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "columns": list(df.columns),
        "data": df.fillna("").to_dict(orient="records"),
        "total_returned": len(df),
    }
