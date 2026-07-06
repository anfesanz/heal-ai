"""Overall and subgroup performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .calibration import calibration_metrics


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    include_calibration: bool = True,
) -> dict[str, float]:
    """Compute discrimination, error, and calibration metrics."""

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    positive = y_true == 1
    negative = ~positive
    predicted_positive = y_pred == 1
    predicted_negative = ~predicted_positive
    tp = int(np.sum(positive & predicted_positive))
    tn = int(np.sum(negative & predicted_negative))
    fp = int(np.sum(negative & predicted_positive))
    fn = int(np.sum(positive & predicted_negative))

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    false_positive_rate = fp / (fp + tn) if (fp + tn) else np.nan
    false_negative_rate = fn / (fn + tp) if (fn + tp) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) else 0.0
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan

    metrics = {
        "n": len(y_true),
        "outcome_rate": float(np.mean(y_true)),
        "auc": auc,
        "accuracy": (tp + tn) / len(y_true) if len(y_true) else np.nan,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "precision": precision,
        "f1": f1,
    }
    if not include_calibration:
        return metrics

    calibration = calibration_metrics(y_true, y_prob)
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
    threshold: float = 0.5,
    predicted_probabilities_by_model: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Evaluate fitted models on a shared test set."""

    rows = []
    for model_name, model in models.items():
        y_prob = (
            predicted_probabilities_by_model[model_name]
            if predicted_probabilities_by_model is not None
            else model.predict_proba(X_test)[:, 1]
        )
        row = {"model": model_name}
        row.update(classification_metrics(y_test, y_prob, threshold=threshold))
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
) -> pd.DataFrame:
    """Evaluate metrics within each subgroup level."""

    frame = pd.DataFrame({"y_true": y_true.to_numpy(), "y_prob": y_prob, "subgroup": subgroup.to_numpy()})
    rows = []
    for value, group in frame.dropna(subset=["subgroup"]).groupby("subgroup", observed=True):
        row = {
            "model": model_name,
            "subgroup_variable": subgroup_variable,
            "subgroup": value,
        }
        if len(group) < min_n or group["y_true"].nunique() < 2:
            row.update({"n": len(group), "warning": "Too few observations or only one outcome class."})
        else:
            row.update(classification_metrics(group["y_true"], group["y_prob"], threshold=threshold))
            row["warning"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_all_subgroups(
    models: dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    subgroup_variables: list[str],
    threshold: float = 0.5,
    min_n: int = 30,
    predicted_probabilities_by_model: dict[str, np.ndarray] | None = None,
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
                    threshold=threshold,
                    min_n=min_n,
                )
            )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
