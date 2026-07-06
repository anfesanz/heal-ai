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


overall = load_table("overall_model_performance.csv")
subgroup = load_table("subgroup_model_performance.csv")
gaps = load_table("fairness_gap_summary.csv")
calibration = load_table("calibration_metrics.csv")

st.title("HEAL-AI Eval")
st.caption("Health AI evaluation across demographic groups using public NHANES data")

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

metric_cols = ["auc", "sensitivity", "specificity", "false_negative_rate", "brier_score"]
metric_values = selected_overall.iloc[0][metric_cols]
metric_cards = st.columns(len(metric_cols))
for col, metric in zip(metric_cards, metric_cols):
    col.metric(metric.replace("_", " ").title(), f"{metric_values[metric]:.3f}")

tab_metrics, tab_calibration, tab_gaps = st.tabs(["Subgroups", "Calibration", "Gaps"])

with tab_metrics:
    st.subheader("Subgroup performance")
    display_cols = [
        "subgroup",
        "n",
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
    st.dataframe(selected_gaps.reset_index(drop=True), width="stretch")
    gap_path = FIGURE_DIR / "09_performance_gap_plot.png"
    if gap_path.exists():
        st.image(str(gap_path), caption="Best-worst subgroup gaps")
