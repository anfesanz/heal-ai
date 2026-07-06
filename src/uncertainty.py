"""Bootstrap uncertainty intervals for model metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .evaluation import classification_metrics


BOOTSTRAP_METRICS = [
    "auc",
    "sensitivity",
    "specificity",
    "false_positive_rate",
    "false_negative_rate",
    "brier_score",
]


def _bootstrap_metrics_fast(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Compute only the metrics needed for bootstrap intervals."""

    y_pred = y_prob >= threshold
    positive = y_true == 1
    negative = ~positive
    tp = np.sum(positive & y_pred)
    tn = np.sum(negative & ~y_pred)
    fp = np.sum(negative & y_pred)
    fn = np.sum(positive & ~y_pred)
    return {
        "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "sensitivity": tp / (tp + fn) if (tp + fn) else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else np.nan,
        "false_negative_rate": fn / (fn + tp) if (fn + tp) else np.nan,
        "brier_score": float(np.mean((y_true - y_prob) ** 2)),
    }


def bootstrap_metric_intervals(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metrics: list[str] | None = None,
    n_bootstrap: int = 500,
    random_state: int = 42,
    min_n: int = 30,
) -> pd.DataFrame:
    """Estimate 95% bootstrap confidence intervals for selected metrics."""

    metrics = metrics or BOOTSTRAP_METRICS
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    if len(y_true) < min_n or len(np.unique(y_true)) < 2:
        return pd.DataFrame(
            {
                "metric": metrics,
                "estimate": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "warning": "Too few observations or only one outcome class.",
            }
        )

    rng = np.random.default_rng(random_state)
    point = classification_metrics(y_true, y_prob, include_calibration=False)
    samples = {metric: [] for metric in metrics}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot = _bootstrap_metrics_fast(y_true[idx], y_prob[idx])
        for metric in metrics:
            samples[metric].append(boot[metric])

    rows = []
    for metric in metrics:
        values = np.asarray(samples[metric], dtype=float)
        values = values[~np.isnan(values)]
        rows.append(
            {
                "metric": metric,
                "estimate": point.get(metric, np.nan),
                "ci_low": np.percentile(values, 2.5) if len(values) else np.nan,
                "ci_high": np.percentile(values, 97.5) if len(values) else np.nan,
                "warning": "" if len(values) else "Bootstrap failed to produce stable samples.",
            }
        )
    return pd.DataFrame(rows)


def bootstrap_by_subgroup(
    y_true: pd.Series,
    y_prob: np.ndarray,
    subgroup: pd.Series,
    model_name: str,
    subgroup_variable: str,
    n_bootstrap: int = 500,
    min_n: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    """Bootstrap intervals within subgroup levels."""

    frame = pd.DataFrame({"y_true": y_true.to_numpy(), "y_prob": y_prob, "subgroup": subgroup.to_numpy()})
    rows = []
    for i, (value, group) in enumerate(frame.dropna(subset=["subgroup"]).groupby("subgroup", observed=True)):
        intervals = bootstrap_metric_intervals(
            group["y_true"].to_numpy(),
            group["y_prob"].to_numpy(),
            n_bootstrap=n_bootstrap,
            random_state=random_state + i,
            min_n=min_n,
        )
        intervals.insert(0, "subgroup", value)
        intervals.insert(0, "subgroup_variable", subgroup_variable)
        intervals.insert(0, "model", model_name)
        rows.append(intervals)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
