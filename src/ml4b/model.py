from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline


@dataclass
class SleepModelBundle:
    model: Pipeline
    feature_columns: list[str]
    metrics: dict[str, float]
    metadata: dict[str, Any]


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    exclude = {"night_id", "window_start", "window_end", "label", "sleep_fraction", "sample_count"}
    return [column for column in frame.columns if column not in exclude and pd.api.types.is_numeric_dtype(frame[column])]


def create_model_pipeline(random_state: int = 42) -> Pipeline:
    classifier = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
        max_depth=None,
        min_samples_leaf=2,
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", classifier),
    ])


def train_sleep_model(feature_frame: pd.DataFrame, group_column: str = "night_id", random_state: int = 42) -> SleepModelBundle:
    if feature_frame.empty:
        raise ValueError("Feature frame is empty.")

    usable = feature_frame.loc[feature_frame["label"].isin(["AWAKE", "SLEEP"])].copy()
    if usable.empty:
        raise ValueError("No labeled windows are available for training.")

    feature_columns = _feature_columns(usable)
    if not feature_columns:
        raise ValueError("No numeric feature columns were found.")

    X = usable[feature_columns]
    y = (usable["label"] == "SLEEP").astype(int)
    groups = usable[group_column] if group_column in usable.columns else pd.Series(np.arange(len(usable)))

    model = create_model_pipeline(random_state=random_state)
    unique_groups = pd.Series(groups).dropna().unique()
    metrics: dict[str, float] = {}

    if len(unique_groups) >= 2:
        n_splits = min(5, len(unique_groups))
        cv = GroupKFold(n_splits=n_splits)
        scoring = {
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "f1": "f1",
            "precision": "precision",
            "recall": "recall",
            "roc_auc": "roc_auc",
        }
        cv_result = cross_validate(model, X, y, groups=groups, cv=cv, scoring=scoring, error_score="raise")
        for name, values in cv_result.items():
            if name.startswith("test_"):
                metrics[name.replace("test_", "")] = float(np.nanmean(values))

    model.fit(X, y)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    metrics.update(
        {
            "train_accuracy": float(accuracy_score(y, predictions)),
            "train_balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
            "train_f1": float(f1_score(y, predictions, zero_division=0)),
            "train_precision": float(precision_score(y, predictions, zero_division=0)),
            "train_recall": float(recall_score(y, predictions, zero_division=0)),
        }
    )
    try:
        metrics["train_roc_auc"] = float(roc_auc_score(y, probabilities))
    except ValueError:
        metrics["train_roc_auc"] = float("nan")

    metadata = {
        "n_samples": int(len(usable)),
        "n_groups": int(len(unique_groups)),
        "feature_count": int(len(feature_columns)),
    }
    return SleepModelBundle(model=model, feature_columns=feature_columns, metrics=metrics, metadata=metadata)


def predict_sleep_probability(bundle: SleepModelBundle, feature_frame: pd.DataFrame) -> pd.DataFrame:
    if feature_frame.empty:
        return pd.DataFrame()

    missing_columns = [column for column in bundle.feature_columns if column not in feature_frame.columns]
    if missing_columns:
        raise ValueError(f"Feature frame is missing required columns: {missing_columns}")

    predictions = feature_frame.copy()
    probabilities = bundle.model.predict_proba(predictions[bundle.feature_columns])[:, 1]
    predictions["sleep_probability"] = probabilities
    predictions["predicted_label"] = np.where(predictions["sleep_probability"] >= 0.5, "SLEEP", "AWAKE")
    return predictions


def save_model_bundle(bundle: SleepModelBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": bundle.model,
            "feature_columns": bundle.feature_columns,
            "metrics": bundle.metrics,
            "metadata": bundle.metadata,
        },
        path,
    )


def load_model_bundle(path: Path) -> SleepModelBundle:
    payload = joblib.load(path)
    return SleepModelBundle(
        model=payload["model"],
        feature_columns=payload["feature_columns"],
        metrics=payload.get("metrics", {}),
        metadata=payload.get("metadata", {}),
    )
