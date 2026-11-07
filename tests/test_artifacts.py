from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"

EXPECTED_COLUMNS = {
    "overall_model_performance.csv": {
        "model",
        "auc",
        "sensitivity",
        "specificity",
        "false_negative_rate",
        "brier_score",
    },
    "subgroup_model_performance.csv": {
        "model",
        "subgroup_variable",
        "subgroup",
        "n",
        "auc",
    },
    "fairness_gap_summary.csv": {
        "model",
        "subgroup_variable",
        "metric",
        "gap",
    },
    "calibration_metrics.csv": {
        "model",
        "subgroup_variable",
        "subgroup",
        "brier_score",
    },
}


def test_dashboard_tables_have_required_columns_and_rows() -> None:
    for filename, required_columns in EXPECTED_COLUMNS.items():
        path = TABLE_DIR / filename
        assert path.is_file(), f"Missing dashboard table: {path}"

        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames is not None
            assert required_columns.issubset(reader.fieldnames)
            assert next(reader, None) is not None, f"Dashboard table is empty: {path}"
