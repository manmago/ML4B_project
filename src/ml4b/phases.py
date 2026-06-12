"""Multiclass sleep-phase model (Awake / Light / Deep / REM).

Mirrors the structure of :mod:`ml4b.model` but predicts the four Samsung Health sleep
stages instead of binary AWAKE/SLEEP, and is trained on real ground-truth stage labels
(see :mod:`ml4b.samsung`) rather than heuristic pseudo-labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import feature_columns as _feature_columns

STAGE_COLUMN = "stage"


@dataclass
class PhaseModelBundle:
    model: Pipeline
    feature_columns: list[str]
    classes: list[str]
    metrics: dict[str, float]
    metadata: dict[str, Any]


def create_phase_pipeline(random_state: int = 42) -> Pipeline:
    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def train_phase_model(feature_frame: pd.DataFrame, group_column: str = "night_id", random_state: int = 42) -> PhaseModelBundle:
    if feature_frame.empty:
        raise ValueError("Feature frame is empty.")
    if STAGE_COLUMN not in feature_frame.columns:
        raise ValueError(f"Feature frame is missing the '{STAGE_COLUMN}' label column.")

    usable = feature_frame.loc[feature_frame[STAGE_COLUMN].notna()].copy()
    if usable.empty:
        raise ValueError("No stage-labeled windows are available for training.")

    feature_names = _feature_columns(usable)
    if not feature_names:
        raise ValueError("No numeric feature columns were found.")

    X = usable[feature_names]
    y = usable[STAGE_COLUMN].astype(str)
    groups = usable[group_column] if group_column in usable.columns else pd.Series(np.arange(len(usable)))

    classes = sorted(y.unique().tolist())
    model = create_phase_pipeline(random_state=random_state)
    unique_groups = pd.Series(groups).dropna().unique()
    metrics: dict[str, float] = {}
    metadata: dict[str, Any] = {
        "n_samples": int(len(usable)),
        "n_groups": int(len(unique_groups)),
        "feature_count": int(len(feature_names)),
        "classes": classes,
        "class_counts": {str(stage): int(count) for stage, count in y.value_counts().items()},
        "per_night_counts": {
            str(night): int(count) for night, count in usable[group_column].value_counts().items()
        }
        if group_column in usable.columns
        else {},
    }

    if len(unique_groups) >= 2:
        n_splits = min(5, len(unique_groups))
        cv = GroupKFold(n_splits=n_splits)
        oof_pred = cross_val_predict(model, X, y, groups=groups, cv=cv)
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y, oof_pred))
        metrics["macro_f1"] = float(f1_score(y, oof_pred, average="macro", labels=classes, zero_division=0))
        metrics["weighted_f1"] = float(f1_score(y, oof_pred, average="weighted", labels=classes, zero_division=0))
        report = classification_report(y, oof_pred, labels=classes, output_dict=True, zero_division=0)
        for stage in classes:
            metrics[f"{stage}_precision"] = float(report[stage]["precision"])
            metrics[f"{stage}_recall"] = float(report[stage]["recall"])
            metrics[f"{stage}_f1"] = float(report[stage]["f1-score"])
        metadata["confusion_matrix"] = confusion_matrix(y, oof_pred, labels=classes).tolist()
        metadata["confusion_labels"] = classes
        metadata["cv_splits"] = int(n_splits)

    model.fit(X, y)
    train_pred = model.predict(X)
    metrics["train_balanced_accuracy"] = float(balanced_accuracy_score(y, train_pred))
    metrics["train_macro_f1"] = float(f1_score(y, train_pred, average="macro", labels=classes, zero_division=0))

    return PhaseModelBundle(
        model=model,
        feature_columns=feature_names,
        classes=classes,
        metrics=metrics,
        metadata=metadata,
    )


def predict_phase_stages(bundle: PhaseModelBundle, feature_frame: pd.DataFrame) -> pd.DataFrame:
    if feature_frame.empty:
        return pd.DataFrame()

    missing_columns = [column for column in bundle.feature_columns if column not in feature_frame.columns]
    if missing_columns:
        raise ValueError(f"Feature frame is missing required columns: {missing_columns}")

    predictions = feature_frame.copy()
    predicted = bundle.model.predict(predictions[bundle.feature_columns])
    predictions["stage_name"] = [str(label) for label in predicted]
    return predictions


def save_phase_model_bundle(bundle: PhaseModelBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": bundle.model,
            "feature_columns": bundle.feature_columns,
            "classes": bundle.classes,
            "metrics": bundle.metrics,
            "metadata": bundle.metadata,
        },
        path,
    )


def load_phase_model_bundle(path: Path) -> PhaseModelBundle:
    payload = joblib.load(path)
    return PhaseModelBundle(
        model=payload["model"],
        feature_columns=payload["feature_columns"],
        classes=payload.get("classes", []),
        metrics=payload.get("metrics", {}),
        metadata=payload.get("metadata", {}),
    )
