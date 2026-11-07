"""Run the full HEAL-AI Eval workflow and save outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import load_or_build_nhanes_dataset
from .evaluation import evaluate_all_subgroups, evaluate_models
from .fairness_metrics import subgroup_gap_summary
from .modelling import choose_thresholds, save_models, train_models
from .plotting import (
    plot_bootstrap_forest,
    plot_calibration_by_model,
    plot_calibration_by_subgroup,
    plot_decision_curve,
    plot_gap_summary,
    plot_outcome_distribution,
    plot_roc_curves,
    plot_subgroup_metric,
)
from .preprocessing import GROUP_COLUMNS, TARGET
from .uncertainty import bootstrap_by_subgroup, bootstrap_metric_intervals


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
WEIGHT_COLUMN = "WTMEC2YR"


def outcome_definition_by_subgroup(df: pd.DataFrame, subgroup_variables: list[str]) -> pd.DataFrame:
    """Compare self-reported and HbA1c-inclusive diabetes prevalence by subgroup.

    Shows how much of each group's diabetes is undiagnosed, i.e. missed by a
    self-report-only label. Weighted columns use NHANES exam weights.
    """

    needed = {"diabetes", "diabetes_self_report", "undiagnosed_diabetes"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()
    weights = df[WEIGHT_COLUMN] if WEIGHT_COLUMN in df else pd.Series(np.nan, index=df.index)

    def summarise(frame: pd.DataFrame, w: pd.Series) -> dict[str, float]:
        row = {
            "n": len(frame),
            "n_cases": int(frame["diabetes"].sum()),
            "prevalence_self_report": frame["diabetes_self_report"].mean(),
            "prevalence_self_report_or_hba1c": frame["diabetes"].mean(),
            "undiagnosed_share_of_cases": frame.loc[frame["diabetes"] == 1, "undiagnosed_diabetes"].mean(),
        }
        valid = w.notna() & (w > 0) & frame["diabetes_self_report"].notna()
        if valid.any():
            ww = w[valid]
            sub = frame[valid]
            cases = sub["diabetes"] == 1
            row["weighted_prevalence_self_report"] = np.average(sub["diabetes_self_report"], weights=ww)
            row["weighted_prevalence_self_report_or_hba1c"] = np.average(sub["diabetes"], weights=ww)
            row["weighted_undiagnosed_share_of_cases"] = (
                np.average(sub.loc[cases, "undiagnosed_diabetes"], weights=ww[cases]) if cases.any() else np.nan
            )
        return row

    rows = [{"subgroup_variable": "Overall", "subgroup": "Overall", **summarise(df, weights)}]
    for variable in subgroup_variables:
        for value, group in df.dropna(subset=[variable]).groupby(variable, observed=True):
            rows.append({"subgroup_variable": variable, "subgroup": value, **summarise(group, weights.loc[group.index])})
    return pd.DataFrame(rows)


def run(
    download: bool = False,
    n_bootstrap: int = 300,
    rebuild: bool = False,
    target_sensitivity: float = 0.80,
) -> None:
    """Train models, evaluate them, and write portfolio-ready artifacts."""

    analysis_df = load_or_build_nhanes_dataset(download=download, rebuild=rebuild)
    subgroup_variables_present = [col for col in GROUP_COLUMNS if col in analysis_df.columns]
    models, X_train, X_test, y_train, y_test = train_models(analysis_df)
    thresholds = choose_thresholds(analysis_df, X_train, y_train, target_sensitivity=target_sensitivity)
    predicted_probabilities_by_model = {
        name: model.predict_proba(X_test)[:, 1] for name, model in models.items()
    }
    test_weights = None
    if WEIGHT_COLUMN in analysis_df:
        candidate = analysis_df.loc[X_test.index, WEIGHT_COLUMN]
        if candidate.notna().all() and (candidate > 0).all():
            test_weights = candidate.to_numpy(dtype=float)

    table_dir = OUTPUTS / "tables"
    figure_dir = OUTPUTS / "figures"
    model_dir = OUTPUTS / "models"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "model": list(thresholds),
            "threshold": list(thresholds.values()),
            "target_sensitivity": target_sensitivity,
            "selection": "5-fold out-of-fold predictions on the training set",
        }
    ).to_csv(table_dir / "decision_thresholds.csv", index=False)

    outcome_table = outcome_definition_by_subgroup(analysis_df, subgroup_variables_present)
    if not outcome_table.empty:
        outcome_table.to_csv(table_dir / "outcome_definition_by_subgroup.csv", index=False)

    overall = evaluate_models(
        models,
        X_test,
        y_test,
        threshold=thresholds,
        predicted_probabilities_by_model=predicted_probabilities_by_model,
    )
    overall.to_csv(table_dir / "overall_model_performance.csv", index=False)

    subgroup_metrics = evaluate_all_subgroups(
        models,
        X_test,
        y_test,
        subgroup_variables_present,
        threshold=thresholds,
        predicted_probabilities_by_model=predicted_probabilities_by_model,
    )
    subgroup_metrics.to_csv(table_dir / "subgroup_model_performance.csv", index=False)

    calibration_columns = [
        "model",
        "subgroup_variable",
        "subgroup",
        "n",
        "brier_score",
        "calibration_slope",
        "calibration_intercept",
        "warning",
    ]
    overall_calibration = overall.copy()
    overall_calibration["subgroup_variable"] = "Overall"
    overall_calibration["subgroup"] = "Overall"
    overall_calibration["warning"] = ""
    calibration_table = pd.concat(
        [
            overall_calibration[[col for col in calibration_columns if col in overall_calibration]],
            subgroup_metrics[[col for col in calibration_columns if col in subgroup_metrics]],
        ],
        ignore_index=True,
    )
    calibration_table.to_csv(
        table_dir / "calibration_metrics.csv",
        index=False,
    )

    gaps = subgroup_gap_summary(subgroup_metrics)
    gaps.to_csv(table_dir / "fairness_gap_summary.csv", index=False)

    # Survey-weighted sensitivity analysis: same models and thresholds, metrics
    # weighted by NHANES exam weights (point estimates only; no design-based CIs).
    if test_weights is not None:
        evaluate_models(
            models,
            X_test,
            y_test,
            threshold=thresholds,
            predicted_probabilities_by_model=predicted_probabilities_by_model,
            sample_weight=test_weights,
        ).to_csv(table_dir / "overall_model_performance_weighted.csv", index=False)
        subgroup_weighted = evaluate_all_subgroups(
            models,
            X_test,
            y_test,
            subgroup_variables_present,
            threshold=thresholds,
            predicted_probabilities_by_model=predicted_probabilities_by_model,
            sample_weight=test_weights,
        )
        subgroup_weighted.to_csv(table_dir / "subgroup_model_performance_weighted.csv", index=False)
        subgroup_gap_summary(subgroup_weighted).to_csv(table_dir / "fairness_gap_summary_weighted.csv", index=False)

    best_model_name = overall.iloc[0]["model"]
    best_model = models[best_model_name]
    best_model_probabilities = predicted_probabilities_by_model[best_model_name]
    best_threshold = thresholds[best_model_name]
    intervals = []
    overall_intervals = bootstrap_metric_intervals(
        y_test.to_numpy(),
        best_model_probabilities,
        n_bootstrap=n_bootstrap,
        threshold=best_threshold,
    )
    overall_intervals.insert(0, "subgroup", "Overall")
    overall_intervals.insert(0, "subgroup_variable", "Overall")
    overall_intervals.insert(0, "model", best_model_name)
    intervals.append(overall_intervals)
    for subgroup_variable in subgroup_variables_present:
        intervals.append(
            bootstrap_by_subgroup(
                y_test,
                best_model_probabilities,
                X_test[subgroup_variable],
                best_model_name,
                subgroup_variable,
                n_bootstrap=n_bootstrap,
                threshold=best_threshold,
            )
        )
    bootstrap_table = pd.concat(intervals, ignore_index=True)
    bootstrap_table.to_csv(table_dir / "bootstrap_uncertainty_intervals.csv", index=False)

    if subgroup_variables_present:
        plot_outcome_distribution(
            analysis_df,
            TARGET,
            subgroup_variables_present[0],
            figure_dir / "01_outcome_distribution_by_subgroup.png",
        )
    plot_roc_curves(
        models,
        X_test,
        y_test,
        figure_dir / "02_overall_roc_curves.png",
        predicted_probabilities_by_model=predicted_probabilities_by_model,
    )
    plot_calibration_by_model(
        models,
        X_test,
        y_test,
        figure_dir / "03_calibration_curve_by_model.png",
        predicted_probabilities_by_model=predicted_probabilities_by_model,
    )
    if subgroup_variables_present:
        plot_calibration_by_subgroup(
            best_model,
            X_test,
            y_test,
            subgroup_variables_present[0],
            figure_dir / "04_calibration_curve_by_subgroup.png",
            y_prob=best_model_probabilities,
        )
    plot_subgroup_metric(subgroup_metrics, "auc", best_model_name, figure_dir / "05_subgroup_auc_comparison.png")
    plot_subgroup_metric(
        subgroup_metrics,
        "false_negative_rate",
        best_model_name,
        figure_dir / "06_subgroup_false_negative_rate_comparison.png",
    )
    plot_subgroup_metric(
        subgroup_metrics,
        "brier_score",
        best_model_name,
        figure_dir / "07_subgroup_brier_score_comparison.png",
    )
    plot_bootstrap_forest(bootstrap_table, "auc", figure_dir / "08_bootstrap_ci_forest_plot.png")
    plot_gap_summary(gaps, best_model_name, figure_dir / "09_performance_gap_plot.png")
    plot_decision_curve(
        y_test,
        predicted_probabilities_by_model,
        figure_dir / "10_decision_curve.png",
        operating_points=thresholds,
    )
    save_models(models, model_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HEAL-AI Eval analysis.")
    parser.add_argument("--download", action="store_true", help="Download public CDC NHANES files if needed.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the processed dataset from the raw NHANES files (downloads missing files).",
    )
    parser.add_argument("--n-bootstrap", type=int, default=300, help="Bootstrap resamples.")
    parser.add_argument(
        "--target-sensitivity",
        type=float,
        default=0.80,
        help="Sensitivity each model's decision threshold is chosen to reach on training data.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        download=args.download,
        n_bootstrap=args.n_bootstrap,
        rebuild=args.rebuild,
        target_sensitivity=args.target_sensitivity,
    )
