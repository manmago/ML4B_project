from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from .config import ACCEL_PHASE_DIR, DEFAULT_STEP_SECONDS, DEFAULT_WINDOW_SECONDS, FEATURE_CACHE_DIR, MODELS_DIR, PROCESSED_DIR, RAW_DIR, SAMSUNG_SLEEP_PATH, SLEEPDATA_DIR
from .features import WindowSpec, build_feature_dataset, build_feature_dataset_for_prediction, extract_window_features_with_labels
from .io import discover_night_dirs, load_huawei_sleep_intervals, load_manual_labels, load_sensor_csv
from .model import SleepModelBundle, predict_sleep_probability, save_model_bundle, train_sleep_model
from .phases import PhaseModelBundle, predict_phase_stages, save_phase_model_bundle, train_phase_model
from .preprocess import NightFrameBundle, add_signal_magnitudes, build_labeled_night_frames, build_or_load_night_frame, merge_sensor_streams
from .samsung import assign_window_stage_labels, load_samsung_sleep_intervals


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


# ---------------------------------------------------------------------------
# Sleep-phase (multiclass) pipeline
# ---------------------------------------------------------------------------
def discover_accelerometer_night_files(accel_dir: Path = ACCEL_PHASE_DIR) -> list[Path]:
    if not accel_dir.exists():
        return []
    return sorted(accel_dir.glob("*.csv"), key=lambda path: path.name)


def build_phase_night_frame(accel_csv: Path) -> pd.DataFrame:
    """Load a single accelerometer-only night CSV into a merged, magnitude-enriched frame."""
    accel = load_sensor_csv(accel_csv, "accelerometer")
    merged = merge_sensor_streams(accel, pd.DataFrame())
    if merged.empty:
        return merged
    enriched = add_signal_magnitudes(merged)
    enriched["night_id"] = accel_csv.stem
    return enriched


def build_phase_feature_dataset(
    accel_dir: Path = ACCEL_PHASE_DIR,
    samsung_path: Path = SAMSUNG_SLEEP_PATH,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    step_seconds: int = DEFAULT_STEP_SECONDS,
) -> pd.DataFrame:
    """Build a stage-labeled feature table across all accelerometer nights.

    Each night CSV is windowed with the shared ml4b feature extractor (unlabeled), then every
    window is stamped with the Samsung Health stage it overlaps the most. Windows with no
    Samsung overlap are dropped, so the result contains only ground-truth-labeled windows.
    """
    intervals = load_samsung_sleep_intervals(samsung_path)
    spec = WindowSpec(window_seconds=window_seconds, step_seconds=step_seconds)
    frames: list[pd.DataFrame] = []
    for accel_csv in discover_accelerometer_night_files(accel_dir):
        night_frame = build_phase_night_frame(accel_csv)
        if night_frame.empty:
            continue
        features = extract_window_features_with_labels(night_frame, window_spec=spec, require_labels=False)
        if features.empty:
            continue
        labeled = assign_window_stage_labels(features, intervals)
        if not labeled.empty:
            frames.append(labeled)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def train_phase_workspace_model(
    accel_dir: Path = ACCEL_PHASE_DIR,
    samsung_path: Path = SAMSUNG_SLEEP_PATH,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    step_seconds: int = DEFAULT_STEP_SECONDS,
    model_path: Path | None = None,
) -> tuple[pd.DataFrame, PhaseModelBundle]:
    feature_frame = build_phase_feature_dataset(
        accel_dir=accel_dir,
        samsung_path=samsung_path,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
    )
    model_bundle = train_phase_model(feature_frame)
    if model_path is not None:
        save_phase_model_bundle(model_bundle, model_path)
    return feature_frame, model_bundle


def predict_phase_hypnogram_for_night(
    night_frame: pd.DataFrame,
    model_bundle: PhaseModelBundle,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    step_seconds: int = DEFAULT_STEP_SECONDS,
) -> pd.DataFrame:
    feature_frame = build_feature_dataset_for_prediction(
        [night_frame], window_spec=WindowSpec(window_seconds=window_seconds, step_seconds=step_seconds)
    )
    if feature_frame.empty:
        return pd.DataFrame()
    return predict_phase_stages(model_bundle, feature_frame)


def build_samsung_binary_feature_dataset(
    accel_dir: Path = ACCEL_PHASE_DIR,
    samsung_path: Path = SAMSUNG_SLEEP_PATH,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    step_seconds: int = DEFAULT_STEP_SECONDS,
) -> pd.DataFrame:
    """Stage-labeled accelerometer windows collapsed to a binary AWAKE/SLEEP `label` column.

    Reuses the Samsung-labeled phase dataset and maps `Awake -> AWAKE`, everything else
    (`Light`/`Deep`/`REM`) `-> SLEEP`, so the existing binary trainer (`train_sleep_model`)
    can be applied to real smartwatch wake/sleep ground truth.
    """
    frame = build_phase_feature_dataset(
        accel_dir=accel_dir,
        samsung_path=samsung_path,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
    )
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["label"] = frame["stage"].where(frame["stage"] == "Awake", "SLEEP").replace({"Awake": "AWAKE"})
    return frame


def train_samsung_binary_workspace_model(
    accel_dir: Path = ACCEL_PHASE_DIR,
    samsung_path: Path = SAMSUNG_SLEEP_PATH,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    step_seconds: int = DEFAULT_STEP_SECONDS,
    model_path: Path | None = None,
) -> tuple[pd.DataFrame, SleepModelBundle]:
    feature_frame = build_samsung_binary_feature_dataset(
        accel_dir=accel_dir,
        samsung_path=samsung_path,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
    )
    model_bundle = train_sleep_model(feature_frame)
    if model_path is not None:
        save_model_bundle(model_bundle, model_path)
    return feature_frame, model_bundle
