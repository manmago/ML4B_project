from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_TIMEZONE
from .io import load_huawei_sleep_intervals, load_manual_labels


@dataclass(frozen=True)
class NightLabelSource:
    source: str
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]]


def _overlap_seconds(start_a: pd.Timestamp, end_a: pd.Timestamp, start_b: pd.Timestamp, end_b: pd.Timestamp) -> float:
    latest_start = max(start_a, start_b)
    earliest_end = min(end_a, end_b)
    return max(0.0, (earliest_end - latest_start).total_seconds())


def resolve_label_source_for_night(
    night_id: str,
    night_start: pd.Timestamp | None,
    night_end: pd.Timestamp | None,
    manual_labels: pd.DataFrame,
    sleep_intervals: pd.DataFrame,
) -> NightLabelSource:
    manual_row = manual_labels.loc[manual_labels["night_id"] == night_id]
    manual_intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    if not manual_row.empty:
        bed_time = manual_row.iloc[0]["bed_time"]
        wake_time = manual_row.iloc[0]["wake_time"]
        if pd.notna(bed_time) and pd.notna(wake_time):
            manual_intervals.append((bed_time, wake_time))

    huawei_intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    if not sleep_intervals.empty and night_start is not None and night_end is not None:
        for _, row in sleep_intervals.iterrows():
            start_time = row["start_time"]
            end_time = row["end_time"]
            if _overlap_seconds(start_time, end_time, night_start, night_end) > 0:
                huawei_intervals.append((start_time, end_time))

    if huawei_intervals:
        return NightLabelSource(source="huawei", intervals=huawei_intervals)
    if manual_intervals:
        return NightLabelSource(source="manual", intervals=manual_intervals)
    return NightLabelSource(source="none", intervals=[])


def apply_binary_labels(frame: pd.DataFrame, intervals: list[tuple[pd.Timestamp, pd.Timestamp]]) -> pd.DataFrame:
    labeled = frame.copy()
    labeled["label"] = "UNKNOWN"

    if labeled.empty or "time_ns" not in labeled.columns:
        return labeled

    if not intervals:
        return labeled

    labeled["label"] = "AWAKE"
    time_ns = pd.to_numeric(labeled["time_ns"], errors="coerce").to_numpy(dtype="float64")
    for start_time, end_time in intervals:
        start_ns = pd.Timestamp(start_time).value
        end_ns = pd.Timestamp(end_time).value
        mask = (time_ns >= start_ns) & (time_ns <= end_ns)
        labeled.loc[mask, "label"] = "SLEEP"
    return labeled


def load_label_sources(raw_labels_path: Path, sleepdata_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_manual_labels(raw_labels_path), load_huawei_sleep_intervals(sleepdata_dir)


def label_balance(frame: pd.DataFrame) -> dict[str, int]:
    if "label" not in frame.columns:
        return {}
    counts = frame["label"].value_counts(dropna=False)
    return {str(index): int(value) for index, value in counts.items()}
