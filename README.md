# HEAL-AI Eval: Health AI Evaluation Across Demographic Groups

HEAL-AI Eval shows how average model performance can hide differences between demographic groups. It uses public CDC NHANES 2017–2018 data to predict diabetes using an outcome based on self-report and HbA1c, with about ten predictors: age, sex, race/ethnicity, education, income, BMI, blood pressure, cholesterol, and smoking. The project trains logistic regression, random forest, and XGBoost models, then evaluates them overall and by age, sex, education, poverty level, and race/ethnicity. Evaluation covers discrimination, error rates, Brier score, calibration slope and intercept, best-versus-worst subgroup gaps, and bootstrap confidence intervals for selected metrics.

**Live dashboard:** [demgroupseval.streamlit.app](https://demgroupseval.streamlit.app)

## Project Motivation

The goal is not to build a complex deep learning system. The goal is to show a reproducible workflow for evaluating health prediction models overall and by demographic subgroups, with attention to calibration, subgroup error, performance gaps, and uncertainty.

## Why Health AI Evaluation Matters

Prediction models used in health settings can perform differently across population groups because of data quality, disease prevalence, access to care, measurement differences, and structural inequities. Reporting only average discrimination or accuracy can hide clinically important subgroup failures.

## Dataset

This project is designed for public NHANES data from the CDC. The default workflow can:

- read a cleaned local CSV at `data/raw/nhanes_clean.csv`;
- reuse an analysis-ready file at `data/processed/nhanes_diabetes_analysis.csv`;
- optionally download public CDC NHANES 2017–2018 XPT files with `python -m src.run_analysis --download`.

After changing the cleaning logic, rebuild the cached analysis file with `python -m src.run_analysis --rebuild`.

No private data are used.

## Outcome Definition

The primary outcome (`diabetes`) combines self-report with a laboratory measure:

- `1`: participant reports being told they have diabetes (`DIQ010`), **or** has HbA1c ≥ 6.5% (`LBXGH`, GHB_J file);
- `0`: participant reports no diabetes or borderline diabetes **and** has HbA1c < 6.5%;
- excluded: no self-reported diagnosis and no HbA1c result.

Self-report alone misses undiagnosed diabetes, which is more common in groups with less access to healthcare. In a subgroup fairness analysis, a self-report-only label would bias the very comparisons being made. The self-report-only label (`diabetes_self_report`) and an `undiagnosed_diabetes` flag are kept, and `outcome_definition_by_subgroup.csv` compares both definitions by subgroup. In the 2017–2018 data, about 17% of cases are undiagnosed (survey-weighted), with large differences between groups (for example 28% among non-Hispanic Black adults vs 12% among non-Hispanic White adults).

HbA1c is used only to define the outcome, never as a predictor. This is a demonstration outcome and should not be interpreted as a clinical diagnostic model.

## Predictors

The default predictors include:

- age;
- sex;
- race/ethnicity;
- education;
- family income-to-poverty group;
- BMI;
- systolic and diastolic blood pressure;
- total cholesterol;
- smoking history.

## Models

The project trains three simple supervised learning models:

- logistic regression;
- random forest;
- XGBoost.

All models use scikit-learn pipelines with imputation, categorical encoding, and numeric scaling where appropriate. No class weighting is used, so all three models produce probabilities on the same scale and can be compared fairly.

### Decision threshold

Threshold-based metrics (sensitivity, specificity, false negative rate, etc.) use a cutoff chosen for each model to reach **80% sensitivity**, a screening-oriented operating point. The cutoff is chosen on 5-fold cross-validated predictions from the training set and then applied unchanged to the held-out test set, so test-set sensitivity is an honest estimate. One cutoff is used for everyone (no group-specific thresholds). Thresholds are saved in `decision_thresholds.csv`; change the target with `--target-sensitivity`. A decision curve (net benefit across risk thresholds) is also produced. The "best" model is the one with the highest test-set AUC.

## Evaluation Metrics

Overall model performance includes:

- AUC;
- accuracy;
- sensitivity / recall;
- specificity;
- false positive rate;
- false negative rate;
- precision;
- F1 score;
- Brier score;
- calibration slope;
- calibration intercept.

## Subgroup Evaluation

Reusable subgroup evaluation functions calculate the same metrics by:

- age group;
- sex;
- education;
- income / poverty group;
- race/ethnicity.

This helps demonstrate how average performance can hide weaker performance in specific demographic groups.

Subgroups with fewer than 30 participants or fewer than 20 outcome cases are reported with a warning instead of metrics, because sensitivity and calibration are unstable there.

The gap summary reports the best-worst difference for each metric, labelling best and worst according to each metric's direction: higher is better for AUC and sensitivity; lower is better for false negative rate, Brier score and calibration error. Note that Brier score also depends on prevalence, so it is naturally lower in groups with few cases.

### Survey weights

NHANES uses a complex, multistage sample. As a sensitivity analysis, all overall and subgroup metrics are also computed with NHANES exam weights (`WTMEC2YR`), giving population-weighted point estimates for US adults (`*_weighted.csv` tables and a toggle in the dashboard). Models are trained without weights. Weighted estimates are point estimates only: confidence intervals do not yet account for strata (`SDMVSTRA`) and PSUs (`SDMVPSU`).

## Calibration

The project estimates:

- Brier score overall and by subgroup;
- calibration slope and intercept overall and by subgroup;
- calibration curves overall by model;
- calibration curves by subgroup for the best-performing model.

## Uncertainty Quantification

Bootstrap resampling is used to estimate 95% confidence intervals for key metrics:

- AUC;
- sensitivity;
- specificity;
- false positive rate;
- false negative rate;
- Brier score.

Intervals use the same decision threshold as the point estimates. Small subgroup samples return warnings instead of unstable intervals. The bootstrap resamples individuals and ignores the NHANES survey design.

## Outputs

Tables are saved to `outputs/tables/`:

- `overall_model_performance.csv` and `overall_model_performance_weighted.csv`;
- `subgroup_model_performance.csv` and `subgroup_model_performance_weighted.csv`;
- `calibration_metrics.csv`;
- `fairness_gap_summary.csv` and `fairness_gap_summary_weighted.csv`;
- `bootstrap_uncertainty_intervals.csv`;
- `decision_thresholds.csv`;
- `outcome_definition_by_subgroup.csv`.

Figures are saved to `outputs/figures/`:

- outcome distribution by subgroup;
- overall ROC curves;
- calibration curves;
- subgroup AUC comparison;
- subgroup false negative rate comparison;
- subgroup Brier score comparison;
- bootstrap confidence interval forest plot;
- performance gap plot;
- decision curve.

Fitted models are saved to `outputs/models/`.

## How To Run

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the analysis using a local cleaned NHANES CSV:

```bash
python -m src.run_analysis
```

Or download public CDC NHANES files:

```bash
python -m src.run_analysis --download
```

Launch the dashboard:

```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

Or use the Makefile shortcuts:

```bash
make install
make analysis-download N_BOOTSTRAP=500
make dashboard
```

## Testing

Install the analysis, dashboard and test dependencies, then run the test suite:

```bash
pip install -r requirements-dev.txt
pytest -q
```

GitHub Actions runs the same checks with Python 3.12 for every pull request and every push to `main`.

## Deployment

The interactive dashboard is deployed on Streamlit Community Cloud at <https://demgroupseval.streamlit.app>, using this GitHub repository as its source:

- branch: `main`;
- entry point: `app/streamlit_app.py`;
- Python version: 3.12;
- application secrets: none required.

The dashboard reads the versioned tables and figures in `outputs/`, so it does not retrain models or download data during startup. This keeps the web deployment lightweight and reproducible.

## Project Structure

```text
heal-ai-eval/
├── .github/workflows/
├── .streamlit/
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
├── data/
├── notebooks/
├── src/
├── outputs/
├── tests/
└── app/
```

## Responsible AI Interpretation

This project frames fairness cautiously as subgroup performance assessment. Differences in AUC, false negative rate, Brier score, or calibration error are signals for further investigation, not proof of algorithmic bias by themselves. A responsible analysis should consider sample sizes, missingness, measurement quality, subgroup definitions, clinical context, and the consequences of different error types.

## Limitations

- **Outcome misclassification differs by group.** Self-reported diagnosis misses undiagnosed diabetes, which is more common in groups with less access to care. Adding HbA1c ≥ 6.5% reduces but does not remove this: a single HbA1c measurement is not a clinical diagnosis, and HbA1c can differ across racial/ethnic groups for reasons unrelated to blood glucose.
- Survey-weighted estimates are point estimates; confidence intervals do not account for the NHANES strata and PSUs.
- NHANES is cross-sectional and not designed for individual clinical prediction deployment.
- Subgroup estimates can be unstable for small groups.
- The selected predictors are limited to variables available in the public files.
- This project is a research and portfolio demonstration, not a validated clinical tool.
