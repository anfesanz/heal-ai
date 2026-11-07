"""Calibration metrics and curves."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationWarning
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


EPS = 1e-6


def calibration_slope_intercept(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> tuple[float, float]:
    """Estimate calibration slope and intercept via logistic recalibration.

    When `sample_weight` is given (e.g. NHANES survey weights), a weighted
    logistic fit gives population-weighted point estimates.
    """

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob), EPS, 1 - EPS)
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    logits = np.log(y_prob / (1 - y_prob))
    design = sm.add_constant(logits)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PerfectSeparationWarning)
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            if sample_weight is None:
                result = sm.Logit(y_true, design).fit(disp=False, maxiter=100)
                if not result.mle_retvals.get("converged", False):
                    return np.nan, np.nan
            else:
                weights = np.asarray(sample_weight, dtype=float)
                weights = weights / weights.mean()
                result = sm.GLM(y_true, design, family=sm.families.Binomial(), var_weights=weights).fit(maxiter=100)
                if not result.converged:
                    return np.nan, np.nan
        return float(result.params[1]), float(result.params[0])
    except Exception:
        return np.nan, np.nan


def calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    """Return Brier score, slope, and intercept (optionally survey-weighted)."""

    slope, intercept = calibration_slope_intercept(y_true, y_prob, sample_weight=sample_weight)
    return {
        "brier_score": brier_score_loss(y_true, y_prob, sample_weight=sample_weight),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }


def calibration_curve_frame(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    model_name: str | None = None,
    subgroup: str | None = None,
) -> pd.DataFrame:
    """Return a tidy calibration curve data frame."""

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    frame = pd.DataFrame({"mean_predicted_probability": prob_pred, "observed_probability": prob_true})
    if model_name is not None:
        frame["model"] = model_name
    if subgroup is not None:
        frame["subgroup"] = subgroup
    return frame
