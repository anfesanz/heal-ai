# HEAL-AI Eval: Health AI Evaluation Across Demographic Groups

Health AI models are often evaluated using average performance metrics. However, average accuracy can conceal systematic differences in performance across demographic groups. This project demonstrates a responsible AI evaluation workflow for population health prediction models using NHANES public data.

## Project Motivation

The goal is not to build a complex deep learning system. The goal is to show a clear, reproducible workflow for evaluating health prediction models overall and by demographic subgroups, with attention to calibration, subgroup error, performance gaps, and uncertainty.

## Why Health AI Evaluation Matters

Prediction models used in health settings can perform differently across population groups because of data quality, disease prevalence, access to care, measurement differences, and structural inequities. Reporting only average discrimination or accuracy can hide clinically important subgroup failures.

## Dataset

This project is designed for public NHANES data from the CDC. The default workflow can:

- read a cleaned local CSV at `data/raw/nhanes_clean.csv`;
- reuse an analysis-ready file at `data/processed/nhanes_diabetes_analysis.csv`;
- optionally download public CDC NHANES XPT files with `python -m src.run_analysis --download`.

No private data are used.

## Outcome Definition

The implemented outcome is binary self-reported diabetes status from NHANES diabetes questionnaire data:

- `1`: participant reports being told they have diabetes;
- `0`: participant reports no diabetes or borderline diabetes.

This is a demonstration outcome and should not be interpreted as a clinical diagnostic model.

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

All models use scikit-learn pipelines with imputation, categorical encoding, and numeric scaling where appropriate.

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

Small subgroup samples return warnings instead of unstable intervals.

## Outputs

Tables are saved to `outputs/tables/`:

- `overall_model_performance.csv`;
- `subgroup_model_performance.csv`;
- `calibration_metrics.csv`;
- `fairness_gap_summary.csv`;
- `bootstrap_uncertainty_intervals.csv`.

Figures are saved to `outputs/figures/`:

- outcome distribution by subgroup;
- overall ROC curves;
- calibration curves;
- subgroup AUC comparison;
- subgroup false negative rate comparison;
- subgroup Brier score comparison;
- bootstrap confidence interval forest plot;
- performance gap plot.

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
streamlit run app/streamlit_app.py
```

Or use the Makefile shortcuts:

```bash
make install
make analysis-download N_BOOTSTRAP=500
make dashboard
```

## Project Structure

```text
heal-ai-eval/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
├── notebooks/
├── src/
├── outputs/
└── app/
```

## Responsible AI Interpretation

This project frames fairness cautiously as subgroup performance assessment. Differences in AUC, false negative rate, Brier score, or calibration error are signals for further investigation, not proof of algorithmic bias by themselves. A responsible analysis should consider sample sizes, missingness, measurement quality, subgroup definitions, clinical context, and the consequences of different error types.

## Limitations

- NHANES is cross-sectional and not designed for individual clinical prediction deployment.
- Self-reported outcomes may contain measurement error.
- Subgroup estimates can be unstable for small groups.
- The selected predictors are limited to variables available in the public files.
