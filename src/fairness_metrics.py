"""Subgroup performance gap analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


GAP_METRICS = [
    "auc",
    "sensitivity",
    "false_negative_rate",
    "brier_score",
    "calibration_error",
]


def subgroup_gap_summary(
    subgroup_metrics: pd.DataFrame,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Calculate best-worst subgroup gaps for each model and subgroup variable."""

    metrics = metrics or GAP_METRICS
    rows = []
    valid = subgroup_metrics[subgroup_metrics.get("warning", "").fillna("") == ""]
    for (model, subgroup_variable), group in valid.groupby(["model", "subgroup_variable"]):
        for metric in metrics:
            if metric not in group:
                continue
            values = group[metric].dropna()
            if values.empty:
                gap = np.nan
                best = worst = None
            else:
                gap = values.max() - values.min()
                best = group.loc[group[metric].idxmax(), "subgroup"]
                worst = group.loc[group[metric].idxmin(), "subgroup"]
            rows.append(
                {
                    "model": model,
                    "subgroup_variable": subgroup_variable,
                    "metric": metric,
                    "gap": gap,
                    "best_subgroup": best,
                    "worst_subgroup": worst,
                }
            )
    return pd.DataFrame(rows)
