"""Streamlit dashboard for HEAL-AI Eval."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"


st.set_page_config(page_title="HEAL-AI Eval", layout="wide")


@st.cache_data
def load_table(name: str) -> pd.DataFrame:
    path = TABLE_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


calibration = load_table("calibration_metrics.csv")
thresholds = load_table("decision_thresholds.csv")
outcome_definition = load_table("outcome_definition_by_subgroup.csv")
has_weighted = not load_table("overall_model_performance_weighted.csv").empty

st.title("HEAL-AI Eval")
st.caption("Health AI evaluation across demographic groups using public NHANES data")

weighting = "Unweighted (study sample)"
if has_weighted:
    weighting = st.radio(
        "Estimates",
        ["Unweighted (study sample)", "Survey-weighted (US adult population)"],
        horizontal=True,
        help="Survey-weighted estimates use NHANES exam weights (WTMEC2YR). They are point estimates only.",
    )
suffix = "_weighted" if weighting.startswith("Survey") else ""
overall = load_table(f"overall_model_performance{suffix}.csv")
subgroup = load_table(f"subgroup_model_performance{suffix}.csv")
gaps = load_table(f"fairness_gap_summary{suffix}.csv")

if overall.empty or subgroup.empty:
    st.warning(
        "No output tables found. Run `python -m src.run_analysis --download` "
        "or add NHANES CSV data and run `python -m src.run_analysis`."
    )
    st.stop()

models = sorted(overall["model"].dropna().unique())
subgroup_variables = sorted(subgroup["subgroup_variable"].dropna().unique())

control_1, control_2 = st.columns(2)
with control_1:
    model_name = st.selectbox("Model", models)
with control_2:
    subgroup_variable = st.selectbox("Subgroup variable", subgroup_variables)

selected_overall = overall[overall["model"] == model_name]
selected_subgroup = subgroup[
    (subgroup["model"] == model_name) & (subgroup["subgroup_variable"] == subgroup_variable)
]
selected_gaps = gaps[(gaps["model"] == model_name) & (gaps["subgroup_variable"] == subgroup_variable)]
selected_calibration = calibration[
    (calibration["model"] == model_name) & (calibration["subgroup_variable"] == subgroup_variable)
]

metric_cols = [col for col in ["threshold", "auc", "sensitivity", "specificity", "false_negative_rate", "brier_score"] if col in selected_overall]
metric_values = selected_overall.iloc[0][metric_cols]
metric_cards = st.columns(len(metric_cols))
for col, metric in zip(metric_cards, metric_cols):
    label = "Decision threshold" if metric == "threshold" else metric.replace("_", " ").title()
    col.metric(label, f"{metric_values[metric]:.3f}")
if not thresholds.empty:
    target = thresholds["target_sensitivity"].iloc[0]
    st.caption(
        f"Each model's threshold was chosen to reach {target:.0%} sensitivity on cross-validated training "
        "predictions, then applied unchanged to the held-out test set. Models are trained without class weighting."
    )

tab_metrics, tab_calibration, tab_gaps, tab_outcome = st.tabs(["Subgroups", "Calibration", "Gaps", "Outcome definition"])

with tab_metrics:
    st.subheader("Subgroup performance")
    display_cols = [
        "subgroup",
        "n",
        "n_positive",
        "outcome_rate",
        "auc",
        "sensitivity",
        "specificity",
        "false_negative_rate",
        "brier_score",
        "calibration_slope",
        "warning",
    ]
    st.dataframe(
        selected_subgroup[[col for col in display_cols if col in selected_subgroup]].reset_index(drop=True),
        width="stretch",
    )

    chart_cols = st.columns(2)
    if "false_negative_rate" in selected_subgroup:
        chart_cols[0].bar_chart(selected_subgroup.set_index("subgroup")["false_negative_rate"])
    if "brier_score" in selected_subgroup:
        chart_cols[1].bar_chart(selected_subgroup.set_index("subgroup")["brier_score"])

with tab_calibration:
    st.subheader("Calibration metrics")
    st.caption("Unweighted test-set estimates.")
    st.dataframe(selected_calibration.reset_index(drop=True), width="stretch")
    calibration_path = FIGURE_DIR / "03_calibration_curve_by_model.png"
    subgroup_calibration_path = FIGURE_DIR / "04_calibration_curve_by_subgroup.png"
    image_cols = st.columns(2)
    if calibration_path.exists():
        image_cols[0].image(str(calibration_path), caption="Overall calibration by model")
    if subgroup_calibration_path.exists():
        image_cols[1].image(str(subgroup_calibration_path), caption="Calibration by subgroup")

with tab_gaps:
    st.subheader("Performance gap summary")
    st.caption(
        "Best and worst follow each metric's direction: higher is better for AUC and sensitivity, lower is better "
        "for false negative rate, Brier score and calibration error. Subgroups with fewer than 20 cases are excluded."
    )
    st.dataframe(selected_gaps.reset_index(drop=True), width="stretch")
    gap_path = FIGURE_DIR / "09_performance_gap_plot.png"
    if gap_path.exists():
        st.image(str(gap_path), caption="Best-worst subgroup gaps")

with tab_outcome:
    st.subheader("Outcome definition")
    st.markdown(
        "The outcome is **self-reported diagnosed diabetes or HbA1c ≥ 6.5%**. Self-report alone misses "
        "undiagnosed diabetes, which is more common in groups with less access to care, so a self-report-only "
        "label can make a model look fairer or less fair than it is. The table compares both definitions and shows "
        "the share of cases that were undiagnosed."
    )
    if outcome_definition.empty:
        st.info("Outcome comparison table not found. Rerun the analysis with `--rebuild`.")
    else:
        selected_outcome = outcome_definition[
            outcome_definition["subgroup_variable"].isin(["Overall", subgroup_variable])
        ].reset_index(drop=True)
        st.dataframe(selected_outcome, width="stretch")
