from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .config import DEFAULT_STEP_SECONDS, DEFAULT_WINDOW_SECONDS
from .preprocess import add_signal_magnitudes


@dataclass(frozen=True)
class WindowSpec:
    window_seconds: int = DEFAULT_WINDOW_SECONDS
    step_seconds: int = DEFAULT_STEP_SECONDS


def feature_columns(frame: pd.DataFrame) -> list[str]:
    exclude = {"night_id", "window_start", "window_end", "label", "sleep_fraction", "sample_count"}
    return [column for column in frame.columns if column not in exclude and pd.api.types.is_numeric_dtype(frame[column])]


def _signal_columns(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "accelerometer_x",
        "accelerometer_y",
        "accelerometer_z",
        "accelerometer_magnitude",
        "gyroscope_x",
        "gyroscope_y",
        "gyroscope_z",
        "gyroscope_magnitude",
    ]
    return [column for column in preferred if column in frame.columns]


def _safe_stat(values: pd.Series, stat: str) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    cleaned = np.asarray(numeric, dtype=float)
    cleaned = cleaned[~np.isnan(cleaned)]
    if cleaned.size == 0:
        return float("nan")
    if stat == "mean":
        return float(np.mean(cleaned))
    if stat == "std":
        return float(np.std(cleaned, ddof=0))
    if stat == "min":
        return float(np.min(cleaned))
    if stat == "max":
        return float(np.max(cleaned))
    if stat == "median":
        return float(np.median(cleaned))
    if stat == "iqr":
        return float(np.percentile(cleaned, 75) - np.percentile(cleaned, 25))
    if stat == "energy":
        return float(np.mean(np.square(cleaned)))
    if stat == "range":
        return float(np.max(cleaned) - np.min(cleaned))
    raise ValueError(f"Unsupported statistic: {stat}")


def extract_window_features(frame: pd.DataFrame, window_spec: WindowSpec | None = None) -> pd.DataFrame:
    return extract_window_features_with_labels(frame, window_spec=window_spec, require_labels=True)


def extract_window_features_with_labels(
    frame: pd.DataFrame,
    window_spec: WindowSpec | None = None,
    require_labels: bool = True,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    spec = window_spec or WindowSpec()
    enriched = add_signal_magnitudes(frame)
    enriched = enriched.sort_values("time_ns").reset_index(drop=True)
    enriched = enriched.dropna(subset=["time_ns"]).copy()
    if enriched.empty:
        return pd.DataFrame()

    signal_columns = _signal_columns(enriched)
    if not signal_columns:
        return pd.DataFrame()

    start_ns = int(enriched["time_ns"].min())
    end_ns = int(enriched["time_ns"].max())
    window_ns = int(spec.window_seconds * 1_000_000_000)
    step_ns = int(spec.step_seconds * 1_000_000_000)

    rows: list[dict[str, object]] = []
    for window_start_ns in range(start_ns, end_ns - window_ns + 1, step_ns):
        window_end_ns = window_start_ns + window_ns
        window = enriched[(enriched["time_ns"] >= window_start_ns) & (enriched["time_ns"] < window_end_ns)]
        if window.empty:
            continue

        known_labels = window.get("label", pd.Series(dtype=str))
        known_labels = known_labels[known_labels.isin(["AWAKE", "SLEEP"])]
        if require_labels and known_labels.empty:
            continue

        sleep_fraction = float((known_labels == "SLEEP").mean()) if not known_labels.empty else float("nan")
        reference_label = None
        if not known_labels.empty:
            reference_label = "SLEEP" if sleep_fraction >= 0.5 else "AWAKE"
        row: dict[str, object] = {
            "night_id": window["night_id"].iloc[0] if "night_id" in window.columns else None,
            "window_start": pd.to_datetime(window_start_ns, unit="ns", utc=True).tz_convert("Europe/Berlin"),
            "window_end": pd.to_datetime(window_end_ns, unit="ns", utc=True).tz_convert("Europe/Berlin"),
            "label": reference_label,
            "sleep_fraction": sleep_fraction,
            "sample_count": int(len(window)),
        }

        for column in signal_columns:
            series = window[column]
            for stat in ["mean", "std", "min", "max", "median", "iqr", "energy", "range"]:
                row[f"{column}_{stat}"] = _safe_stat(series, stat)

        rows.append(row)

    return pd.DataFrame(rows)


def build_feature_dataset(night_frames: Iterable[pd.DataFrame], window_spec: WindowSpec | None = None) -> pd.DataFrame:
    frames = [extract_window_features_with_labels(frame, window_spec=window_spec, require_labels=True) for frame in night_frames if not frame.empty]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_feature_dataset_for_prediction(night_frames: Iterable[pd.DataFrame], window_spec: WindowSpec | None = None) -> pd.DataFrame:
    frames = [extract_window_features_with_labels(frame, window_spec=window_spec, require_labels=False) for frame in night_frames if not frame.empty]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
