from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import DEFAULT_MERGE_TOLERANCE_MS, DEFAULT_TIMEZONE, PREPROCESSED_NIGHTS_DIR
from .io import get_night_time_bounds, load_metadata, load_night_sensor_bundle
from .labels import NightLabelSource, apply_binary_labels, resolve_label_source_for_night


JOBLIB_COMPRESSION_LEVEL = 3


@dataclass(frozen=True)
class NightFrameBundle:
    night_id: str
    frame: pd.DataFrame
    metadata: dict
    label_source: NightLabelSource


def night_cache_path(night_id: str, processed_dir: Path = PREPROCESSED_NIGHTS_DIR) -> Path:
    return processed_dir / f"{night_id}.joblib"


def _prepare_sensor_frame(frame: pd.DataFrame, sensor_name: str) -> pd.DataFrame:
    if frame.empty:
        return frame

    renamed = frame.copy()
    if "time_ns" not in renamed.columns:
        renamed["time_ns"] = pd.to_numeric(renamed["time"], errors="coerce")
    renamed = renamed.sort_values("time_ns").reset_index(drop=True)
    numeric_columns = [column for column in renamed.columns if column.startswith(sensor_name + "_")]
    for column in numeric_columns:
        renamed[column] = pd.to_numeric(renamed[column], errors="coerce")
    return renamed


def merge_sensor_streams(
    accelerometer: pd.DataFrame,
    gyroscope: pd.DataFrame,
    tolerance_ms: int = DEFAULT_MERGE_TOLERANCE_MS,
) -> pd.DataFrame:
    if accelerometer.empty and gyroscope.empty:
        return pd.DataFrame()
    if accelerometer.empty:
        return gyroscope.copy()
    if gyroscope.empty:
        return accelerometer.copy()

    accel = _prepare_sensor_frame(accelerometer, "accelerometer")
    gyro = _prepare_sensor_frame(gyroscope, "gyroscope")
    tolerance = int(tolerance_ms * 1_000_000)

    merged = pd.merge_asof(
        accel.sort_values("time_ns"),
        gyro.sort_values("time_ns"),
        on="time_ns",
        direction="nearest",
        tolerance=tolerance,
        suffixes=("", "_gyro"),
    )

    if "time_gyro" in merged.columns:
        merged["time"] = merged["time"].combine_first(merged["time_gyro"])
        merged = merged.drop(columns=["time_gyro"])

    merged = merged.dropna(subset=[column for column in merged.columns if column.startswith("accelerometer_")]).reset_index(drop=True)
    gyro_columns = [column for column in merged.columns if column.startswith("gyroscope_")]
    if gyro_columns:
        merged = merged.dropna(subset=gyro_columns, how="any").reset_index(drop=True)
    return merged


def prepare_night_frame(
    night_dir: Path,
    manual_labels: pd.DataFrame,
    sleep_intervals: pd.DataFrame,
    merge_tolerance_ms: int = DEFAULT_MERGE_TOLERANCE_MS,
) -> NightFrameBundle:
    night_id = night_dir.name
    bundle = load_night_sensor_bundle(night_dir)
    metadata = load_metadata(night_dir)
    merged = merge_sensor_streams(
        bundle.get("accelerometer", pd.DataFrame()),
        bundle.get("gyroscope", pd.DataFrame()),
        tolerance_ms=merge_tolerance_ms,
    )
    if merged.empty:
        return NightFrameBundle(night_id=night_id, frame=merged, metadata=metadata, label_source=NightLabelSource("none", []))

    night_start, night_end = get_night_time_bounds(bundle)
    label_source = resolve_label_source_for_night(
        night_id=night_id,
        night_start=night_start,
        night_end=night_end,
        manual_labels=manual_labels,
        sleep_intervals=sleep_intervals,
    )
    labeled = apply_binary_labels(merged, label_source.intervals)
    labeled["night_id"] = night_id
    return NightFrameBundle(night_id=night_id, frame=labeled, metadata=metadata, label_source=label_source)


def save_night_frame_cache(bundle: NightFrameBundle, processed_dir: Path = PREPROCESSED_NIGHTS_DIR) -> Path:
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_path = night_cache_path(bundle.night_id, processed_dir)
    joblib.dump(bundle, cache_path, compress=JOBLIB_COMPRESSION_LEVEL)
    return cache_path


def load_cached_night_frame(night_id: str, processed_dir: Path = PREPROCESSED_NIGHTS_DIR) -> NightFrameBundle | None:
    cache_path = night_cache_path(night_id, processed_dir)
    if not cache_path.exists():
        return None
    payload = joblib.load(cache_path)
    if isinstance(payload, NightFrameBundle):
        return payload
    return None


def build_or_load_night_frame(
    night_dir: Path,
    manual_labels: pd.DataFrame,
    sleep_intervals: pd.DataFrame,
    merge_tolerance_ms: int = DEFAULT_MERGE_TOLERANCE_MS,
    processed_dir: Path = PREPROCESSED_NIGHTS_DIR,
    force_rebuild: bool = False,
) -> NightFrameBundle:
    if not force_rebuild:
        cached = load_cached_night_frame(night_dir.name, processed_dir=processed_dir)
        if cached is not None:
            return cached

    bundle = prepare_night_frame(night_dir, manual_labels, sleep_intervals, merge_tolerance_ms=merge_tolerance_ms)
    if not bundle.frame.empty:
        save_night_frame_cache(bundle, processed_dir=processed_dir)
    return bundle


def build_labeled_night_frames(
    raw_dir: Path,
    manual_labels: pd.DataFrame,
    sleep_intervals: pd.DataFrame,
    merge_tolerance_ms: int = DEFAULT_MERGE_TOLERANCE_MS,
    processed_dir: Path = PREPROCESSED_NIGHTS_DIR,
    force_rebuild: bool = False,
) -> list[NightFrameBundle]:
    night_dirs = [path for path in raw_dir.iterdir() if path.is_dir() and path.name.count("_") == 1]
    night_dirs = sorted(night_dirs, key=lambda path: path.name)
    bundles: list[NightFrameBundle] = []
    for night_dir in night_dirs:
        bundle = build_or_load_night_frame(
            night_dir,
            manual_labels,
            sleep_intervals,
            merge_tolerance_ms=merge_tolerance_ms,
            processed_dir=processed_dir,
            force_rebuild=force_rebuild,
        )
        if not bundle.frame.empty:
            bundles.append(bundle)
    return bundles


def add_signal_magnitudes(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    accel_columns = [column for column in ["accelerometer_x", "accelerometer_y", "accelerometer_z"] if column in enriched.columns]
    gyro_columns = [column for column in ["gyroscope_x", "gyroscope_y", "gyroscope_z"] if column in enriched.columns]

    if len(accel_columns) == 3:
        enriched["accelerometer_magnitude"] = np.sqrt(
            enriched["accelerometer_x"].pow(2) + enriched["accelerometer_y"].pow(2) + enriched["accelerometer_z"].pow(2)
        )
    if len(gyro_columns) == 3:
        enriched["gyroscope_magnitude"] = np.sqrt(
            enriched["gyroscope_x"].pow(2) + enriched["gyroscope_y"].pow(2) + enriched["gyroscope_z"].pow(2)
        )
    return enriched
