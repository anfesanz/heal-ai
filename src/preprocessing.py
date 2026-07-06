"""Feature preprocessing utilities."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "diabetes"
GROUP_COLUMNS = ["age_group", "sex", "education", "poverty_group", "race_ethnicity"]
DEFAULT_FEATURES = [
    "age",
    "age_group",
    "sex",
    "race_ethnicity",
    "education",
    "poverty_group",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "total_cholesterol",
    "ever_smoked_100_cigarettes",
]


def split_feature_types(df: pd.DataFrame, features: list[str] | None = None) -> tuple[list[str], list[str]]:
    """Return numeric and categorical feature names."""

    features = features or DEFAULT_FEATURES
    numeric = [col for col in features if pd.api.types.is_numeric_dtype(df[col])]
    categorical = [col for col in features if col not in numeric]
    return numeric, categorical


def build_preprocessor(
    df: pd.DataFrame,
    features: list[str] | None = None,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Build a sklearn preprocessor for numeric and categorical predictors."""

    numeric_features, categorical_features = split_feature_types(df, features)
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def prepare_xy(
    df: pd.DataFrame,
    target: str = TARGET,
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return predictor matrix and binary target."""

    features = features or DEFAULT_FEATURES
    missing = set([target, *features]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    clean = df.dropna(subset=[target]).copy()
    return clean[features], clean[target].astype(int)
