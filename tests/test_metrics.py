from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation import classification_metrics, evaluate_by_subgroup
from src.fairness_metrics import subgroup_gap_summary
from src.modelling import threshold_for_sensitivity


def _subgroup_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": "m",
            "subgroup_variable": "age_group",
            "subgroup": ["young", "middle", "old"],
            "auc": [0.70, 0.80, 0.90],
            "sensitivity": [0.10, 0.60, 0.95],
            "false_negative_rate": [0.90, 0.40, 0.05],
            "brier_score": [0.05, 0.10, 0.20],
            "calibration_error": [0.30, 0.10, 0.20],
            "warning": "",
        }
    )


def test_gap_labels_follow_metric_direction() -> None:
    gaps = subgroup_gap_summary(_subgroup_table()).set_index("metric")

    assert gaps.loc["auc", ["best_subgroup", "worst_subgroup"]].tolist() == ["old", "young"]
    assert gaps.loc["sensitivity", ["best_subgroup", "worst_subgroup"]].tolist() == ["old", "young"]
    # Lower is better: the group that misses most cases must be "worst".
    assert gaps.loc["false_negative_rate", ["best_subgroup", "worst_subgroup"]].tolist() == ["old", "young"]
    assert gaps.loc["brier_score", ["best_subgroup", "worst_subgroup"]].tolist() == ["young", "old"]
    assert gaps.loc["calibration_error", ["best_subgroup", "worst_subgroup"]].tolist() == ["middle", "young"]
    assert gaps.loc["false_negative_rate", "higher_is_better"] == False  # noqa: E712
    assert gaps.loc["false_negative_rate", "gap"] == pytest.approx(0.85)


def test_threshold_reaches_target_sensitivity() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    prob = np.clip(0.3 * y + rng.uniform(0, 0.7, 500), 0, 1)

    threshold = threshold_for_sensitivity(y, prob, target=0.80)
    sensitivity = np.mean(prob[y == 1] >= threshold)

    assert sensitivity >= 0.80
    # Highest such cutoff: nudging it up drops sensitivity below the target.
    assert np.mean(prob[y == 1] >= np.nextafter(threshold, 1)) < 0.80


def test_weighted_metrics_equal_replicated_rows() -> None:
    y = np.array([1, 1, 0, 0, 1, 0])
    prob = np.array([0.9, 0.3, 0.2, 0.6, 0.7, 0.1])
    weights = np.array([2, 1, 3, 1, 1, 2])

    weighted = classification_metrics(y, prob, threshold=0.5, sample_weight=weights)
    replicated = classification_metrics(np.repeat(y, weights), np.repeat(prob, weights), threshold=0.5)

    for metric in ["outcome_rate", "auc", "sensitivity", "specificity", "brier_score"]:
        assert weighted[metric] == pytest.approx(replicated[metric])
    assert weighted["n"] == len(y)


def test_subgroups_with_few_cases_are_flagged() -> None:
    y = pd.Series([1] * 5 + [0] * 95)
    prob = np.linspace(0, 1, 100)
    result = evaluate_by_subgroup(y, prob, pd.Series(["g"] * 100), "m", "var", min_positives=20)

    assert result.loc[0, "warning"].startswith("Only 5 outcome cases")
    assert "sensitivity" not in result or pd.isna(result.loc[0, "sensitivity"])
