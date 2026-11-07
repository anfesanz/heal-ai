"""Overall and subgroup performance metrics."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .calibration import calibration_metrics


Thresholds = float | Mapping[str, float]


def _threshold_for(thresholds: Thresholds, model_name: str) -> float:
    if isinstance(thresholds, Mapping):
        return float(thresholds[model_name])
    return float(thresholds)


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    include_calibration: bool = True,
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute discrimination, error, and calibration metrics.

    With `sample_weight` (NHANES survey weights) every rate is a weighted
    point estimate; `n` always reports the unweighted number of participants.
    """

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    weights = np.ones(len(y_true)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    positive = y_true == 1
    negative = ~positive
    predicted_positive = y_pred == 1
    predicted_negative = ~predicted_positive
    tp = float(np.sum(weights[positive & predicted_positive]))
    tn = float(np.sum(weights[negative & predicted_negative]))
    fp = float(np.sum(weights[negative & predicted_positive]))
    fn = float(np.sum(weights[positive & predicted_negative]))
    total = tp + tn + fp + fn

    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    false_positive_rate = fp / (fp + tn) if (fp + tn) else np.nan
    false_negative_rate = fn / (fn + tp) if (fn + tp) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if not np.isnan(precision) and not np.isnan(sensitivity) and (precision + sensitivity)
        else np.nan
    )
    auc = roc_auc_score(y_true, y_prob, sample_weight=sample_weight) if len(np.unique(y_true)) > 1 else np.nan

    metrics = {
        "n": len(y_true),
        "n_positive": int(positive.sum()),
        "threshold": threshold,
        "outcome_rate": float(np.average(y_true, weights=weights)) if len(y_true) else np.nan,
        "auc": auc,
        "accuracy": (tp + tn) / total if total else np.nan,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "precision": precision,
        "f1": f1,
    }
    if not include_calibration:
        return metrics

    calibration = calibration_metrics(y_true, y_prob, sample_weight=sample_weight)
    calibration["calibration_error"] = (
        abs(calibration["calibration_slope"] - 1) + abs(calibration["calibration_intercept"])
        if not np.isnan(calibration["calibration_slope"]) and not np.isnan(calibration["calibration_intercept"])
        else np.nan
    )
    metrics.update(calibration)
    return metrics


def evaluate_models(
    models: dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: Thresholds = 0.5,
    predicted_probabilities_by_model: dict[str, np.ndarray] | None = None,
    sample_weight: np.ndarray | None = None,
) -> pd.DataFrame:
    """Evaluate fitted models on a shared test set.

    `threshold` is either one cutoff for all models or a {model: cutoff} dict.
    """

    rows = []
    for model_name, model in models.items():
        y_prob = (
            predicted_probabilities_by_model[model_name]
            if predicted_probabilities_by_model is not None
            else model.predict_proba(X_test)[:, 1]
        )
        row = {"model": model_name}
        row.update(
            classification_metrics(
                y_test,
                y_prob,
                threshold=_threshold_for(threshold, model_name),
                sample_weight=sample_weight,
            )
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("auc", ascending=False)


def evaluate_by_subgroup(
    y_true: pd.Series,
    y_prob: np.ndarray,
    subgroup: pd.Series,
    model_name: str,
    subgroup_variable: str,
    threshold: float = 0.5,
    min_n: int = 30,
    min_positives: int = 20,
    sample_weight: np.ndarray | None = None,
) -> pd.DataFrame:
    """Evaluate metrics within each subgroup level.

    Subgroups with fewer than `min_n` participants or fewer than
    `min_positives` outcome cases are reported with a warning instead of
    metrics, because sensitivity and calibration are unstable there.
    """

    frame = pd.DataFrame({"y_true": y_true.to_numpy(), "y_prob": y_prob, "subgroup": subgroup.to_numpy()})
    frame["weight"] = 1.0 if sample_weight is None else np.asarray(sample_weight, dtype=float)
    rows = []
    for value, group in frame.dropna(subset=["subgroup"]).groupby("subgroup", observed=True):
        row = {
            "model": model_name,
            "subgroup_variable": subgroup_variable,
            "subgroup": value,
        }
        n_positive = int(group["y_true"].sum())
        if len(group) < min_n or group["y_true"].nunique() < 2:
            row.update({"n": len(group), "n_positive": n_positive, "warning": "Too few observations or only one outcome class."})
        elif n_positive < min_positives:
            row.update(
                {
                    "n": len(group),
                    "n_positive": n_positive,
                    "outcome_rate": float(np.average(group["y_true"], weights=group["weight"])),
                    "warning": f"Only {n_positive} outcome cases (<{min_positives}); metrics not reported.",
                }
            )
        else:
            row.update(
                classification_metrics(
                    group["y_true"],
                    group["y_prob"],
                    threshold=threshold,
                    sample_weight=None if sample_weight is None else group["weight"].to_numpy(),
                )
            )
            row["warning"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_all_subgroups(
    models: dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    subgroup_variables: list[str],
    threshold: Thresholds = 0.5,
    min_n: int = 30,
    min_positives: int = 20,
    predicted_probabilities_by_model: dict[str, np.ndarray] | None = None,
    sample_weight: np.ndarray | None = None,
) -> pd.DataFrame:
    """Evaluate all models across all requested subgroup variables."""

    frames = []
    for model_name, model in models.items():
        y_prob = (
            predicted_probabilities_by_model[model_name]
            if predicted_probabilities_by_model is not None
            else model.predict_proba(X_test)[:, 1]
        )
        for subgroup_variable in subgroup_variables:
            frames.append(
                evaluate_by_subgroup(
                    y_test,
                    y_prob,
                    X_test[subgroup_variable],
                    model_name,
                    subgroup_variable,
                    threshold=_threshold_for(threshold, model_name),
                    min_n=min_n,
                    min_positives=min_positives,
                    sample_weight=sample_weight,
                )
            )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
