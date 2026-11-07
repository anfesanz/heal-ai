"""Streamlit dashboard for HEAL-AI Eval."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

# --- Display names -----------------------------------------------------------

MODEL_LABELS = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
}
VARIABLE_LABELS = {
    "Overall": "Overall",
    "age_group": "Age group",
    "sex": "Sex",
    "education": "Education",
    "poverty_group": "Income (% of poverty line)",
    "race_ethnicity": "Race/ethnicity",
}
# kind: "percent" values are proportions shown as %; "ratio" are shown as decimals.
METRICS = {
    "threshold": {
        "label": "Decision threshold",
        "kind": "ratio",
        "higher_is_better": None,
        "help": "Predicted risk at or above which a person is flagged as likely to have diabetes.",
    },
    "auc": {
        "label": "AUC",
        "kind": "ratio",
        "higher_is_better": True,
        "help": "How well the model ranks people with diabetes above people without it "
        "(0.5 = chance, 1 = perfect). Higher is better.",
    },
    "sensitivity": {
        "label": "Sensitivity",
        "kind": "percent",
        "higher_is_better": True,
        "help": "Share of people with diabetes whom the model flags. Higher is better.",
    },
    "specificity": {
        "label": "Specificity",
        "kind": "percent",
        "higher_is_better": True,
        "help": "Share of people without diabetes whom the model correctly does not flag. Higher is better.",
    },
    "false_negative_rate": {
        "label": "Missed cases",
        "kind": "percent",
        "higher_is_better": False,
        "help": "False negative rate: share of people with diabetes whom the model misses. Lower is better.",
    },
    "brier_score": {
        "label": "Brier score",
        "kind": "ratio",
        "higher_is_better": False,
        "help": "Average squared error of the predicted risks. Lower is better. It also depends on how common "
        "diabetes is in the group, so compare groups with care.",
    },
    "calibration_error": {
        "label": "Calibration error",
        "kind": "ratio",
        "higher_is_better": False,
        "help": "|slope − 1| + |intercept|: how far predicted risks are from observed risks. Lower is better.",
    },
}
WEIGHTING_OPTIONS = ["Study sample (unweighted)", "US adult population (survey-weighted)"]


def model_label(name: str) -> str:
    return MODEL_LABELS.get(name, name.replace("_", " ").capitalize())


def variable_label(name: str) -> str:
    return VARIABLE_LABELS.get(name, name.replace("_", " ").capitalize())


def format_value(metric: str, value: float) -> str:
    if pd.isna(value):
        return "–"
    if METRICS[metric]["kind"] == "percent":
        return f"{value:.1%}"
    return f"{value:.3f}"


def format_delta(metric: str, value: float, reference: float) -> str | None:
    if pd.isna(value) or pd.isna(reference):
        return None
    diff = value - reference
    if METRICS[metric]["kind"] == "percent":
        return f"{diff * 100:+.1f} pts vs sample"
    return f"{diff:+.3f} vs sample"


def note_text(warning: object) -> str:
    return "⚠ Too few cases to evaluate" if isinstance(warning, str) and warning.strip() else ""


def show_table(
    table: pd.DataFrame,
    percent: tuple[str, ...] = (),
    decimals3: tuple[str, ...] = (),
    decimals2: tuple[str, ...] = (),
    counts: tuple[str, ...] = (),
    help_text: dict[str, str] | None = None,
) -> None:
    """Render a table with consistent number formats, "–" for missing values, and header tooltips.

    Numbers are pre-formatted as right-aligned text because st.dataframe shows
    missing numeric values as "None".
    """

    formats = {col: "{:.1f}%" for col in percent}
    formats |= {col: "{:.3f}" for col in decimals3}
    formats |= {col: "{:.2f}" for col in decimals2}
    formats |= {col: "{:,.0f}" for col in counts}
    display = table.copy()
    help_text = help_text or {}
    column_config = {}
    for col in display.columns:
        if col in formats:
            display[col] = display[col].map(lambda v, f=formats[col]: "–" if pd.isna(v) else f.format(v))
            column_config[col] = st.column_config.TextColumn(col, help=help_text.get(col), alignment="right")
        elif col in help_text:
            column_config[col] = st.column_config.TextColumn(col, help=help_text[col])
    st.dataframe(display, hide_index=True, width="stretch", column_config=column_config)


# --- Data loading ------------------------------------------------------------


@st.cache_data
def _read_csv(path: str, modified: float) -> pd.DataFrame:
    # `modified` is part of the cache key, so a redeployed file is always re-read.
    return pd.read_csv(path)


def load_table(name: str) -> pd.DataFrame:
    """Load an output table; missing files are not cached."""

    path = TABLE_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return _read_csv(str(path), path.stat().st_mtime)


# --- Page --------------------------------------------------------------------

st.set_page_config(page_title="HEAL-AI Eval", layout="wide")

calibration = load_table("calibration_metrics.csv")
thresholds = load_table("decision_thresholds.csv")
outcome_definition = load_table("outcome_definition_by_subgroup.csv")
overall_unweighted = load_table("overall_model_performance.csv")
has_weighted = not load_table("overall_model_performance_weighted.csv").empty

st.title("HEAL-AI Eval")
st.markdown(
    "How well do diabetes prediction models work **for different groups of US adults**? "
    "Choose a model and a demographic variable to compare performance across groups. "
    "Built on public NHANES 2017–2018 data."
)
control_1, control_2, control_3 = st.columns([1, 1, 1.4])
weighting = WEIGHTING_OPTIONS[0]
if has_weighted:
    with control_3:
        weighting = st.radio(
            "Estimates for",
            WEIGHTING_OPTIONS,
            horizontal=True,
            help="Survey-weighted estimates use NHANES exam weights (WTMEC2YR) so results represent the US "
            "adult population. They are point estimates only.",
        )
weighted = weighting == WEIGHTING_OPTIONS[1]
suffix = "_weighted" if weighted else ""
overall = load_table(f"overall_model_performance{suffix}.csv")
subgroup = load_table(f"subgroup_model_performance{suffix}.csv")
gaps = load_table(f"fairness_gap_summary{suffix}.csv")

if overall.empty or subgroup.empty:
    st.warning(
        "No output tables found. Run `python -m src.run_analysis --download` "
        "or add NHANES CSV data and run `python -m src.run_analysis`."
    )
    st.stop()

# Options are the display labels, mapped back to the names used in the tables.
model_options = {model_label(m): m for m in sorted(overall["model"].dropna().unique(), key=model_label)}
variable_options = {
    variable_label(v): v for v in sorted(subgroup["subgroup_variable"].dropna().unique(), key=variable_label)
}

with control_1:
    model_name = model_options[st.selectbox("Model", list(model_options))]
with control_2:
    subgroup_variable = variable_options[st.selectbox("Compare groups by", list(variable_options))]

selected_overall = overall[overall["model"] == model_name]
selected_subgroup = subgroup[
    (subgroup["model"] == model_name) & (subgroup["subgroup_variable"] == subgroup_variable)
].copy()
selected_gaps = gaps[(gaps["model"] == model_name) & (gaps["subgroup_variable"] == subgroup_variable)].copy()
selected_calibration = calibration[
    (calibration["model"] == model_name) & (calibration["subgroup_variable"] == subgroup_variable)
].copy()

# Headline metrics
st.subheader(f"{model_label(model_name)}: overall performance")
card_metrics = [m for m in ["threshold", "auc", "sensitivity", "specificity", "false_negative_rate", "brier_score"] if m in selected_overall]
row = selected_overall.iloc[0]
reference = overall_unweighted[overall_unweighted["model"] == model_name]
reference_row = reference.iloc[0] if weighted and not reference.empty else None
target_text = ""
if not thresholds.empty:
    target_text = f" It was chosen so the model catches {thresholds['target_sensitivity'].iloc[0]:.0%} of cases in training data."
cards = st.columns(len(card_metrics))
for col, metric in zip(cards, card_metrics):
    info = METRICS[metric]
    delta = None
    if reference_row is not None and metric != "threshold" and metric in reference_row:
        delta = format_delta(metric, row[metric], reference_row[metric])
    col.metric(
        info["label"],
        format_value(metric, row[metric]),
        delta=delta,
        delta_color="normal" if info["higher_is_better"] else "inverse",
        help=info["help"] + (target_text if metric == "threshold" else ""),
        border=True,
    )
if not thresholds.empty:
    st.caption(
        f"Each model's threshold reaches {thresholds['target_sensitivity'].iloc[0]:.0%} sensitivity on cross-validated "
        "training predictions and is applied unchanged to the held-out test set. Models are trained without class weighting."
    )

tab_metrics, tab_calibration, tab_gaps, tab_outcome = st.tabs(
    ["Performance by group", "Calibration", "Gaps between groups", "Outcome definition"]
)

with tab_metrics:
    st.subheader(f"Performance by {variable_label(subgroup_variable).lower()}")
    table = pd.DataFrame(
        {
            "Group": selected_subgroup["subgroup"].astype(str),
            "Participants": selected_subgroup.get("n"),
            "Cases": selected_subgroup.get("n_positive"),
            "Prevalence": selected_subgroup.get("outcome_rate") * 100,
            "AUC": selected_subgroup.get("auc"),
            "Sensitivity": selected_subgroup.get("sensitivity") * 100,
            "Specificity": selected_subgroup.get("specificity") * 100,
            "Missed cases": selected_subgroup.get("false_negative_rate") * 100,
            "Brier score": selected_subgroup.get("brier_score"),
            "Calibration slope": selected_subgroup.get("calibration_slope"),
            "Note": selected_subgroup.get("warning", pd.Series(index=selected_subgroup.index, dtype=object)).map(note_text),
        }
    )
    show_table(
        table,
        percent=("Prevalence", "Sensitivity", "Specificity", "Missed cases"),
        decimals3=("AUC", "Brier score"),
        decimals2=("Calibration slope",),
        counts=("Participants", "Cases"),
        help_text={
            "Participants": "People in this group in the test set.",
            "Cases": "People with diabetes in this group in the test set.",
            "Prevalence": "Share of the group with diabetes.",
            "AUC": METRICS["auc"]["help"],
            "Sensitivity": METRICS["sensitivity"]["help"],
            "Specificity": METRICS["specificity"]["help"],
            "Missed cases": METRICS["false_negative_rate"]["help"],
            "Brier score": METRICS["brier_score"]["help"],
            "Calibration slope": "1 = predicted risks spread as observed; <1 = too extreme.",
            "Note": "Groups with fewer than 20 cases are not evaluated.",
        },
    )
    if (table["Note"] != "").any():
        st.caption("⚠ Groups with fewer than 20 cases in the test set are not evaluated, because estimates would be unreliable.")

    chart_data = table[table["Note"] == ""].set_index("Group")
    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.markdown("**Missed cases by group** (lower is better)")
        st.bar_chart(chart_data["Missed cases"], horizontal=True, x_label="", y_label="Missed cases (%)")
    with chart_cols[1]:
        st.markdown("**Brier score by group** (lower is better)")
        st.bar_chart(chart_data["Brier score"], horizontal=True, x_label="", y_label="Brier score")
        st.caption("Brier score is naturally lower in groups where diabetes is rare.")

with tab_calibration:
    st.subheader("Calibration: do predicted risks match observed risks?")
    st.caption("Study-sample estimates on the test set. Ideal: slope 1 and intercept 0.")
    calibration_table = pd.DataFrame(
        {
            "Group": selected_calibration["subgroup"].astype(str),
            "Participants": selected_calibration.get("n"),
            "Brier score": selected_calibration.get("brier_score"),
            "Calibration slope": selected_calibration.get("calibration_slope"),
            "Calibration intercept": selected_calibration.get("calibration_intercept"),
            "Note": selected_calibration.get("warning", pd.Series(index=selected_calibration.index, dtype=object)).map(note_text),
        }
    )
    show_table(
        calibration_table,
        decimals3=("Brier score",),
        decimals2=("Calibration slope", "Calibration intercept"),
        counts=("Participants",),
        help_text={
            "Brier score": METRICS["brier_score"]["help"],
            "Calibration slope": "1 = ideal; <1 = predictions too extreme; >1 = too modest.",
            "Calibration intercept": "0 = ideal; <0 = risks overestimated; >0 = underestimated.",
        },
    )
    calibration_path = FIGURE_DIR / "03_calibration_curve_by_model.png"
    subgroup_calibration_path = FIGURE_DIR / "04_calibration_curve_by_subgroup.png"
    image_cols = st.columns(2)
    if calibration_path.exists():
        image_cols[0].image(str(calibration_path), caption="Predicted vs observed risk, all models")
    if subgroup_calibration_path.exists():
        image_cols[1].image(str(subgroup_calibration_path), caption="Predicted vs observed risk by age group (best model)")

with tab_gaps:
    st.subheader("Gaps between the best and worst group")
    st.caption(
        "For each metric, the difference between the best- and worst-performing group, following each metric's "
        "direction. Groups with fewer than 20 cases are excluded."
    )

    def gap_text(r: pd.Series) -> str:
        if r["metric"] in METRICS and METRICS[r["metric"]]["kind"] == "percent":
            return f"{r['gap'] * 100:.1f} pts"
        return f"{r['gap']:.3f}"

    gap_table = pd.DataFrame(
        {
            "Metric": selected_gaps["metric"].map(lambda m: METRICS.get(m, {}).get("label", m)),
            "Better when": selected_gaps["metric"].map(
                lambda m: {True: "Higher", False: "Lower"}.get(METRICS.get(m, {}).get("higher_is_better"), "")
            ),
            "Gap": selected_gaps.apply(gap_text, axis=1) if not selected_gaps.empty else pd.Series(dtype=str),
            "Best group": selected_gaps.get("best_subgroup"),
            "Worst group": selected_gaps.get("worst_subgroup"),
        }
    )
    show_table(gap_table)
    gap_path = FIGURE_DIR / "09_performance_gap_plot.png"
    if gap_path.exists():
        st.image(str(gap_path), caption="Best–worst gaps for every variable (best model, study sample)", width=760)

with tab_outcome:
    st.subheader("How the outcome is defined")
    st.markdown(
        "The outcome is **self-reported diagnosed diabetes _or_ HbA1c ≥ 6.5%**. Self-report alone misses "
        "undiagnosed diabetes, which is more common in groups with less access to care, so a self-report-only "
        "label can make a model look fairer or less fair than it is. The table compares both definitions."
    )
    if outcome_definition.empty:
        st.info("Outcome comparison table not found. Rerun the analysis with `--rebuild`.")
    else:
        selected_outcome = outcome_definition[
            outcome_definition["subgroup_variable"].isin(["Overall", subgroup_variable])
        ]
        prefix = "weighted_" if weighted and "weighted_prevalence_self_report" in selected_outcome else ""
        outcome_table = pd.DataFrame(
            {
                "Group": selected_outcome["subgroup"].astype(str),
                "Participants": selected_outcome["n"],
                "Cases": selected_outcome["n_cases"],
                "Self-report only": selected_outcome[f"{prefix}prevalence_self_report"] * 100,
                "Self-report or HbA1c": selected_outcome[f"{prefix}prevalence_self_report_or_hba1c"] * 100,
                "Cases undiagnosed": selected_outcome[f"{prefix}undiagnosed_share_of_cases"] * 100,
            }
        )
        show_table(
            outcome_table,
            percent=("Self-report only", "Self-report or HbA1c", "Cases undiagnosed"),
            counts=("Participants", "Cases"),
            help_text={
                "Participants": "All adults in the analysis dataset.",
                "Self-report only": "Diabetes prevalence counting self-reported diagnoses only.",
                "Self-report or HbA1c": "Diabetes prevalence counting self-report or HbA1c ≥ 6.5%.",
                "Cases undiagnosed": "Share of cases with HbA1c ≥ 6.5% but no self-reported diagnosis.",
            },
        )
        st.caption(
            "Survey-weighted percentages (US adult population)." if prefix else "Study-sample percentages (unweighted)."
        )
