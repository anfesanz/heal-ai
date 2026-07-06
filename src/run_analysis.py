"""Run the full HEAL-AI Eval workflow and save outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data_loader import load_or_build_nhanes_dataset
from .evaluation import evaluate_all_subgroups, evaluate_models
from .fairness_metrics import subgroup_gap_summary
from .modelling import save_models, train_models
from .plotting import (
    plot_bootstrap_forest,
    plot_calibration_by_model,
    plot_calibration_by_subgroup,
    plot_gap_summary,
    plot_outcome_distribution,
    plot_roc_curves,
    plot_subgroup_metric,
)
from .preprocessing import GROUP_COLUMNS, TARGET
from .uncertainty import bootstrap_by_subgroup, bootstrap_metric_intervals


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"


def run(download: bool = False, n_bootstrap: int = 300) -> None:
    """Train models, evaluate them, and write portfolio-ready artifacts."""

    analysis_df = load_or_build_nhanes_dataset(download=download)
    subgroup_variables_present = [col for col in GROUP_COLUMNS if col in analysis_df.columns]
    models, _, X_test, _, y_test = train_models(analysis_df)
    predicted_probabilities_by_model = {
        name: model.predict_proba(X_test)[:, 1] for name, model in models.items()
    }

    table_dir = OUTPUTS / "tables"
    figure_dir = OUTPUTS / "figures"
    model_dir = OUTPUTS / "models"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    overall = evaluate_models(
        models,
        X_test,
        y_test,
        predicted_probabilities_by_model=predicted_probabilities_by_model,
    )
    overall.to_csv(table_dir / "overall_model_performance.csv", index=False)

    subgroup_metrics = evaluate_all_subgroups(
        models,
        X_test,
        y_test,
        subgroup_variables_present,
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

    best_model_name = overall.iloc[0]["model"]
    best_model = models[best_model_name]
    best_model_probabilities = predicted_probabilities_by_model[best_model_name]
    intervals = []
    overall_intervals = bootstrap_metric_intervals(
        y_test.to_numpy(),
        best_model_probabilities,
        n_bootstrap=n_bootstrap,
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
    save_models(models, model_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HEAL-AI Eval analysis.")
    parser.add_argument("--download", action="store_true", help="Download public CDC NHANES files if needed.")
    parser.add_argument("--n-bootstrap", type=int, default=300, help="Bootstrap resamples.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(download=args.download, n_bootstrap=args.n_bootstrap)
