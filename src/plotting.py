"""Plotting functions for project outputs."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import RocCurveDisplay


def _save(fig: plt.Figure, path: str | Path | None) -> plt.Figure:
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


def plot_outcome_distribution(df: pd.DataFrame, outcome: str, subgroup: str, path: str | Path | None = None):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    rates = df.groupby(subgroup, observed=True)[outcome].mean().sort_values()
    counts = df.groupby(subgroup, observed=True)[outcome].size().reindex(rates.index)
    labels = [f"{idx}\n(n={counts.loc[idx]})" for idx in rates.index]
    ax.bar(labels, rates.values, color="#2f6f73")
    ax.set_ylabel("Outcome prevalence")
    ax.set_xlabel(subgroup.replace("_", " ").title())
    ax.set_ylim(0, min(1, max(rates.max() * 1.25, 0.1)))
    ax.set_title(f"{outcome.replace('_', ' ').title()} distribution by subgroup")
    ax.tick_params(axis="x", labelrotation=30)
    return _save(fig, path)


def plot_roc_curves(
    models: dict[str, object],
    X_test,
    y_test,
    path: str | Path | None = None,
    predicted_probabilities_by_model: dict[str, np.ndarray] | None = None,
):
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    for name, model in models.items():
        if predicted_probabilities_by_model is None:
            RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
        else:
            RocCurveDisplay.from_predictions(y_test, predicted_probabilities_by_model[name], name=name, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.6")
    ax.set_title("ROC curves by model")
    return _save(fig, path)


def plot_calibration_by_model(
    models: dict[str, object],
    X_test,
    y_test,
    path: str | Path | None = None,
    predicted_probabilities_by_model: dict[str, np.ndarray] | None = None,
):
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    for name, model in models.items():
        y_prob = (
            predicted_probabilities_by_model[name]
            if predicted_probabilities_by_model is not None
            else model.predict_proba(X_test)[:, 1]
        )
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="quantile")
        ax.plot(prob_pred, prob_true, marker="o", label=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.6", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed outcome probability")
    ax.set_title("Calibration curve by model")
    ax.legend()
    return _save(fig, path)


def plot_calibration_by_subgroup(
    model,
    X_test,
    y_test,
    subgroup: str,
    path: str | Path | None = None,
    y_prob: np.ndarray | None = None,
):
    fig, ax = plt.subplots(figsize=(7, 5.2))
    y_prob = y_prob if y_prob is not None else model.predict_proba(X_test)[:, 1]
    frame = pd.DataFrame({"y_true": y_test.to_numpy(), "y_prob": y_prob, "subgroup": X_test[subgroup].to_numpy()})
    for value, group in frame.dropna(subset=["subgroup"]).groupby("subgroup", observed=True):
        if len(group) < 30 or group["y_true"].nunique() < 2:
            continue
        prob_true, prob_pred = calibration_curve(group["y_true"], group["y_prob"], n_bins=6, strategy="quantile")
        ax.plot(prob_pred, prob_true, marker="o", label=value)
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.6")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed outcome probability")
    ax.set_title(f"Calibration by {subgroup.replace('_', ' ')}")
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_subgroup_metric(metrics: pd.DataFrame, metric: str, model_name: str, path: str | Path | None = None):
    data = metrics[(metrics["model"] == model_name) & (metrics.get("warning", "").fillna("") == "")]
    data = data.sort_values(["subgroup_variable", metric])
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(data))))
    labels = data["subgroup_variable"] + ": " + data["subgroup"].astype(str)
    ax.barh(labels, data[metric], color="#4f7cac")
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} by subgroup ({model_name})")
    return _save(fig, path)


def plot_gap_summary(gaps: pd.DataFrame, model_name: str, path: str | Path | None = None):
    data = gaps[gaps["model"] == model_name].copy()
    data["label"] = data["subgroup_variable"] + ": " + data["metric"]
    data = data.sort_values("gap")
    fig, ax = plt.subplots(figsize=(9, max(4, 0.3 * len(data))))
    ax.barh(data["label"], data["gap"], color="#8a5a44")
    ax.set_xlabel("Best-worst subgroup gap")
    ax.set_title(f"Subgroup performance gaps ({model_name})")
    return _save(fig, path)


def plot_bootstrap_forest(intervals: pd.DataFrame, metric: str, path: str | Path | None = None):
    data = intervals[(intervals["metric"] == metric) & (intervals.get("warning", "").fillna("") == "")].copy()
    data["label"] = data["model"] + " | " + data["subgroup_variable"] + ": " + data["subgroup"].astype(str)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(data))))
    y = range(len(data))
    ax.errorbar(
        data["estimate"],
        y,
        xerr=[data["estimate"] - data["ci_low"], data["ci_high"] - data["estimate"]],
        fmt="o",
        color="#2f6f73",
        ecolor="0.55",
    )
    ax.set_yticks(list(y), data["label"])
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(f"Bootstrap 95% intervals for {metric.replace('_', ' ')}")
    return _save(fig, path)
