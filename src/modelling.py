"""Model definitions and training helpers."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .preprocessing import DEFAULT_FEATURES, TARGET, build_preprocessor, prepare_xy


def get_model_specs(random_state: int = 42) -> dict[str, object]:
    """Return the three requested model estimators."""

    try:
        from xgboost import XGBClassifier

        xgb_model = XGBClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
        )
    except ImportError:
        xgb_model = None

    return {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
        "xgboost": xgb_model,
    }


def build_pipeline(
    df: pd.DataFrame,
    model_name: str,
    features: list[str] | None = None,
    random_state: int = 42,
) -> Pipeline:
    """Build a preprocessing-plus-model sklearn pipeline."""

    model = get_model_specs(random_state=random_state)[model_name]
    if model is None:
        raise ImportError("xgboost is not installed. Install requirements.txt to use XGBoost.")
    scale_numeric = model_name == "logistic_regression"
    return Pipeline(
        [
            ("preprocess", build_preprocessor(df, features=features, scale_numeric=scale_numeric)),
            ("model", model),
        ]
    )


def train_models(
    df: pd.DataFrame,
    target: str = TARGET,
    features: list[str] | None = None,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[dict[str, Pipeline], pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Train all models and return fitted pipelines plus the split data."""

    features = features or DEFAULT_FEATURES
    X, y = prepare_xy(df, target=target, features=features)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    models = {}
    for model_name, estimator in get_model_specs(random_state=random_state).items():
        if estimator is None:
            continue
        pipeline = build_pipeline(df, model_name, features=features, random_state=random_state)
        pipeline.fit(X_train, y_train)
        models[model_name] = pipeline
    return models, X_train, X_test, y_train, y_test


def save_models(models: dict[str, Pipeline], output_dir: str | Path) -> None:
    """Persist fitted pipelines."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, output_dir / f"{name}.joblib")
