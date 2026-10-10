"""
CARE-PULSE AI — Experiment Repository
SQLite-backed persistence for experiment runs, model benchmarks, and evaluation metrics.
"""

import json
import sqlite3
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
from app.config import settings

logger = logging.getLogger(__name__)


class ExperimentRepository:
    """Manages SQLite storage for model benchmark experiments and run metadata."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or settings.DB_PATH
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create experiments table if it doesn't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS experiment_runs (
                    run_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    hyperparameters TEXT,
                    metrics TEXT,
                    dataset_fingerprint TEXT,
                    artifact_path TEXT,
                    notes TEXT
                )
                """
            )
            conn.commit()

    def log_run(
        self,
        run_id: str,
        model_name: str,
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, Any],
        dataset_fingerprint: str = "",
        artifact_path: str = "",
        notes: str = ""
    ) -> None:
        """Persist a new experiment record."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO experiment_runs 
                (run_id, model_name, hyperparameters, metrics, dataset_fingerprint, artifact_path, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    model_name,
                    json.dumps(hyperparameters),
                    json.dumps(metrics),
                    dataset_fingerprint,
                    artifact_path,
                    notes
                ),
            )
            conn.commit()
            logger.info(f"Logged experiment run {run_id} ({model_name}) to SQLite repository.")

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific run by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM experiment_runs WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_dict(row)

    def list_runs(self) -> List[Dict[str, Any]]:
        """List all experiment runs ordered by creation timestamp."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM experiment_runs ORDER BY created_at DESC")
            rows = cursor.fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_best_run(self, metric: str = "roc_auc") -> Optional[Dict[str, Any]]:
        """Retrieve best performing run according to the specified metric."""
        runs = self.list_runs()
        if not runs:
            return None
        valid_runs = [r for r in runs if metric in r.get("metrics", {})]
        if not valid_runs:
            return runs[0]
        return max(valid_runs, key=lambda r: r["metrics"].get(metric, 0.0))

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        try:
            data["hyperparameters"] = json.loads(data["hyperparameters"]) if data["hyperparameters"] else {}
        except Exception:
            pass
        try:
            data["metrics"] = json.loads(data["metrics"]) if data["metrics"] else {}
        except Exception:
            pass
        return data


# Global singleton instance
experiment_repo = ExperimentRepository()
