from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from .config import DEFAULT_STEP_SECONDS, DEFAULT_WINDOW_SECONDS, FEATURE_CACHE_DIR, MODELS_DIR, PROCESSED_DIR, RAW_DIR, SLEEPDATA_DIR
from .features import WindowSpec, build_feature_dataset, build_feature_dataset_for_prediction
from .io import discover_night_dirs, load_huawei_sleep_intervals, load_manual_labels
from .model import SleepModelBundle, predict_sleep_probability, save_model_bundle, train_sleep_model
from .preprocess import NightFrameBundle, build_labeled_night_frames, build_or_load_night_frame


@dataclass
class PipelineResult:
    night_bundles: list[NightFrameBundle]
    feature_frame: pd.DataFrame
    model_bundle: SleepModelBundle | None


def feature_cache_path(window_seconds: int, step_seconds: int, processed_dir: Path = PROCESSED_DIR) -> Path:
    return processed_dir / "features" / f"feature_dataset_w{window_seconds}_s{step_seconds}.joblib"


def save_feature_cache(feature_frame: pd.DataFrame, window_seconds: int, step_seconds: int, processed_dir: Path = PROCESSED_DIR) -> Path:
    cache_path = feature_cache_path(window_seconds, step_seconds, processed_dir=processed_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(feature_frame, cache_path)
    return cache_path


def load_feature_cache(window_seconds: int, step_seconds: int, processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame | None:
    cache_path = feature_cache_path(window_seconds, step_seconds, processed_dir=processed_dir)
    if not cache_path.exists():
        return None
    payload = joblib.load(cache_path)
    if isinstance(payload, pd.DataFrame):
        return payload
    return None


def build_workspace_dataset(
    raw_dir: Path = RAW_DIR,
    sleepdata_dir: Path = SLEEPDATA_DIR,
    manual_labels_path: Path | None = None,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    step_seconds: int = DEFAULT_STEP_SECONDS,
    use_cache: bool = True,
    save_cache: bool = True,
) -> PipelineResult:
    if use_cache:
        cached_feature_frame = load_feature_cache(window_seconds, step_seconds)
        if cached_feature_frame is not None:
            return PipelineResult(night_bundles=[], feature_frame=cached_feature_frame, model_bundle=None)

    labels_path = manual_labels_path or (raw_dir / "manual-labels.csv")
    manual_labels = load_manual_labels(labels_path)
    sleep_intervals = load_huawei_sleep_intervals(sleepdata_dir)
    night_bundles = build_labeled_night_frames(raw_dir, manual_labels, sleep_intervals)
    feature_frame = build_feature_dataset(
        [bundle.frame for bundle in night_bundles],
        window_spec=WindowSpec(window_seconds=window_seconds, step_seconds=step_seconds),
    )
    if save_cache and not feature_frame.empty:
        save_feature_cache(feature_frame, window_seconds, step_seconds)
    return PipelineResult(night_bundles=night_bundles, feature_frame=feature_frame, model_bundle=None)


def train_workspace_model(
    raw_dir: Path = RAW_DIR,
    sleepdata_dir: Path = SLEEPDATA_DIR,
    manual_labels_path: Path | None = None,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    step_seconds: int = DEFAULT_STEP_SECONDS,
    model_path: Path | None = None,
    use_cache: bool = True,
) -> tuple[PipelineResult, SleepModelBundle]:
    result = build_workspace_dataset(
        raw_dir=raw_dir,
        sleepdata_dir=sleepdata_dir,
        manual_labels_path=manual_labels_path,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
        use_cache=use_cache,
    )
    model_bundle = train_sleep_model(result.feature_frame)
    if model_path is not None:
        save_model_bundle(model_bundle, model_path)
    result.model_bundle = model_bundle
    return result, model_bundle


def predict_hypnogram_for_night(
    night_frame: pd.DataFrame,
    model_bundle: SleepModelBundle,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    step_seconds: int = DEFAULT_STEP_SECONDS,
) -> pd.DataFrame:
    feature_frame = build_feature_dataset_for_prediction([night_frame], window_spec=WindowSpec(window_seconds=window_seconds, step_seconds=step_seconds))
    if feature_frame.empty:
        return pd.DataFrame()
    return predict_sleep_probability(model_bundle, feature_frame)


def ensure_processed_dir() -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    return PROCESSED_DIR


def list_available_nights(raw_dir: Path = RAW_DIR) -> list[str]:
    return [path.name for path in discover_night_dirs(raw_dir)]


def load_or_build_night_frame(
    night_id: str,
    raw_dir: Path = RAW_DIR,
    sleepdata_dir: Path = SLEEPDATA_DIR,
    manual_labels_path: Path | None = None,
) -> NightFrameBundle:
    labels_path = manual_labels_path or (raw_dir / "manual-labels.csv")
    manual_labels = load_manual_labels(labels_path)
    sleep_intervals = load_huawei_sleep_intervals(sleepdata_dir)
    night_dir = raw_dir / night_id
    return build_or_load_night_frame(night_dir, manual_labels, sleep_intervals)


def preprocess_workspace(
    raw_dir: Path = RAW_DIR,
    sleepdata_dir: Path = SLEEPDATA_DIR,
    manual_labels_path: Path | None = None,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    step_seconds: int = DEFAULT_STEP_SECONDS,
    force_rebuild: bool = False,
) -> PipelineResult:
    labels_path = manual_labels_path or (raw_dir / "manual-labels.csv")
    manual_labels = load_manual_labels(labels_path)
    sleep_intervals = load_huawei_sleep_intervals(sleepdata_dir)
    night_bundles = build_labeled_night_frames(raw_dir, manual_labels, sleep_intervals, force_rebuild=force_rebuild)
    feature_frame = build_feature_dataset(
        [bundle.frame for bundle in night_bundles],
        window_spec=WindowSpec(window_seconds=window_seconds, step_seconds=step_seconds),
    )
    if not feature_frame.empty:
        save_feature_cache(feature_frame, window_seconds, step_seconds)
    return PipelineResult(night_bundles=night_bundles, feature_frame=feature_frame, model_bundle=None)
