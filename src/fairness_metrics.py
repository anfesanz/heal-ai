"""Subgroup performance gap analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


# Metric name -> True when a higher value means better performance.
GAP_METRICS: dict[str, bool] = {
    "auc": True,
    "sensitivity": True,
    "false_negative_rate": False,
    "brier_score": False,
    "calibration_error": False,
}


def subgroup_gap_summary(
    subgroup_metrics: pd.DataFrame,
    metrics: dict[str, bool] | None = None,
) -> pd.DataFrame:
    """Calculate best-worst subgroup gaps for each model and subgroup variable.

    `metrics` maps each metric to whether higher values are better, so that
    "best" and "worst" are labelled correctly for error-type metrics.
    """

    metrics = metrics or GAP_METRICS
    rows = []
    if "warning" in subgroup_metrics:
        valid = subgroup_metrics[subgroup_metrics["warning"].fillna("") == ""]
    else:
        valid = subgroup_metrics
    for (model, subgroup_variable), group in valid.groupby(["model", "subgroup_variable"]):
        for metric, higher_is_better in metrics.items():
            if metric not in group:
                continue
            values = group[metric].dropna()
            if values.empty:
                gap = np.nan
                best = worst = None
            else:
                gap = values.max() - values.min()
                high = group.loc[values.idxmax(), "subgroup"]
                low = group.loc[values.idxmin(), "subgroup"]
                best, worst = (high, low) if higher_is_better else (low, high)
            rows.append(
                {
                    "model": model,
                    "subgroup_variable": subgroup_variable,
                    "metric": metric,
                    "higher_is_better": higher_is_better,
                    "gap": gap,
                    "best_subgroup": best,
                    "worst_subgroup": worst,
                }
            )
    return pd.DataFrame(rows)
