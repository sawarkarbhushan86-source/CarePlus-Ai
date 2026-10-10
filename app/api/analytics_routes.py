"""
CARE-PULSE AI — Healthcare Analytics and Statistical API Routes
"""

import json
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Query, HTTPException

from app.config import settings
from app.services.data_loader import data_loader_service
from app.services.eda_service import eda_service
from app.services.statistics_service import statistics_service

router = APIRouter(prefix="/analytics", tags=["Healthcare Analytics & Statistics"])


@router.get("/overview", summary="Clinical Cohort Overview Statistics")
def get_analytics_overview():
    """Retrieve readmission prevalence, length of stay, and subgroup event rates."""
    eda_file = settings.METRICS_DIR / "eda_summary.json"
    if eda_file.exists():
        try:
            with open(eda_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("overview", {})
        except Exception:
            pass

    try:
        df = data_loader_service.load_dataset(preferred_source="sample", nrows=5000)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return eda_service.compute_overview_stats(df)


@router.get("/odds-ratios", summary="Clinical Risk Factors & Odds Ratios (95% CI)")
def get_odds_ratios():
    """Compute formal Odds Ratios with 95% Wald confidence intervals and p-values."""
    eda_file = settings.METRICS_DIR / "eda_summary.json"
    if eda_file.exists():
        try:
            with open(eda_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("odds_ratios", [])
        except Exception:
            pass

    try:
        df = data_loader_service.load_dataset(preferred_source="sample", nrows=5000)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return statistics_service.compute_all_clinical_odds_ratios(df)


@router.get("/correlations", summary="Clinical Variable Pearson Correlation Matrix")
def get_correlations():
    """Calculate correlation matrix for continuous hospital encounter metrics."""
    try:
        df = data_loader_service.load_dataset(preferred_source="sample", nrows=5000)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return eda_service.compute_correlation_matrix(df)


@router.get("/distribution/{column}", summary="Categorical / Numerical Variable Distribution")
def get_distribution(column: str):
    """Retrieve frequency counts and proportions for a specified feature."""
    try:
        df = data_loader_service.load_dataset(preferred_source="sample", nrows=5000)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return eda_service.compute_distribution(df, column)


@router.get("/hypothesis-test", summary="Chi-Square Independence Test")
def run_hypothesis_test(factor: str = Query(default="diag_1_group")):
    """Run Chi-Square test of independence and compute Cramér's V effect size."""
    try:
        df = data_loader_service.load_dataset(preferred_source="sample", nrows=5000)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return statistics_service.compute_chi_square_test(df, factor)
