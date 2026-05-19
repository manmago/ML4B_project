from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .config import DEFAULT_TIMEZONE, SENSOR_FILE_MAP

NIGHT_FOLDER_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")


def discover_night_dirs(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        return []
    return sorted(
        [path for path in raw_dir.iterdir() if path.is_dir() and NIGHT_FOLDER_PATTERN.match(path.name)],
        key=lambda path: path.name,
    )


def _time_series_to_local_datetime(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return pd.to_datetime(numeric, unit="ns", utc=True).dt.tz_convert(DEFAULT_TIMEZONE)


def _rename_sensor_axes(df: pd.DataFrame, sensor_name: str) -> pd.DataFrame:
    renamed = df.copy()
    axis_columns = [column for column in ["x", "y", "z"] if column in renamed.columns]
    rename_map = {column: f"{sensor_name}_{column}" for column in axis_columns}
    renamed = renamed.rename(columns=rename_map)
    return renamed


def load_sensor_csv(path: Path, sensor_name: str | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    if df.empty or "time" not in df.columns:
        return df

    inferred_name = sensor_name or SENSOR_FILE_MAP.get(path.name, path.stem.lower())
    df["time_ns"] = pd.to_numeric(df["time"], errors="coerce")
    df["time"] = _time_series_to_local_datetime(df["time"])
    df = _rename_sensor_axes(df, inferred_name)
    return df.dropna(subset=["time_ns"]).sort_values("time_ns").reset_index(drop=True)


def load_night_sensor_bundle(night_dir: Path) -> dict[str, pd.DataFrame]:
    bundle: dict[str, pd.DataFrame] = {}
    for file_name, sensor_name in SENSOR_FILE_MAP.items():
        file_path = night_dir / file_name
        if file_path.exists():
            bundle[sensor_name] = load_sensor_csv(file_path, sensor_name)
    return bundle


def load_metadata(night_dir: Path) -> dict[str, Any]:
    metadata_path = night_dir / "Metadata.csv"
    if not metadata_path.exists():
        return {}

    metadata = pd.read_csv(metadata_path)
    if metadata.empty:
        return {}
    return metadata.iloc[0].to_dict()


def load_annotation_notes(night_dir: Path) -> pd.DataFrame:
    annotation_path = night_dir / "Annotation.csv"
    if not annotation_path.exists():
        return pd.DataFrame()
    return pd.read_csv(annotation_path)


def load_manual_labels(labels_path: Path) -> pd.DataFrame:
    if not labels_path.exists():
        return pd.DataFrame(columns=["night_id", "bed_time", "wake_time"])

    labels = pd.read_csv(labels_path)
    labels = labels.replace({"": pd.NA, "nan": pd.NA})

    for column in ["bed_time", "wake_time"]:
        if column not in labels.columns:
            labels[column] = pd.NaT
        parsed = pd.to_datetime(labels[column], errors="coerce")
        if getattr(parsed.dt, "tz", None) is None:
            labels[column] = parsed.dt.tz_localize(DEFAULT_TIMEZONE, nonexistent="shift_forward", ambiguous="NaT")
        else:
            labels[column] = parsed.dt.tz_convert(DEFAULT_TIMEZONE)

    if "night_id" not in labels.columns:
        labels["night_id"] = pd.NA
    return labels[["night_id", "bed_time", "wake_time"]].copy()


def _iter_json_objects(obj: Any) -> Iterable[dict[str, Any]]:
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _iter_json_objects(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_json_objects(item)


def _parse_sleep_interval_payload(payload: dict[str, Any], source_path: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    fields_metadata_raw = payload.get("fieldsMetadata")
    if not fields_metadata_raw:
        return results

    try:
        fields_metadata = json.loads(fields_metadata_raw)
    except json.JSONDecodeError:
        return results

    sleep_info_raw = fields_metadata.get("noonSleepInfo")
    if not sleep_info_raw:
        return results

    try:
        sleep_info = json.loads(sleep_info_raw)
    except json.JSONDecodeError:
        return results

    intervals = sleep_info.get("noonSleepTimeIntervalList", [])
    total_minutes = sleep_info.get("noonSleepTotalTime")
    for interval in intervals:
        start_ms = interval.get("startTime")
        end_ms = interval.get("endTime")
        if start_ms is None or end_ms is None:
            continue
        results.append(
            {
                "start_time": pd.to_datetime(start_ms, unit="ms", utc=True).tz_convert(DEFAULT_TIMEZONE),
                "end_time": pd.to_datetime(end_ms, unit="ms", utc=True).tz_convert(DEFAULT_TIMEZONE),
                "time_zone": interval.get("timeZone", fields_metadata.get("timeZone")),
                "total_minutes": total_minutes,
                "source_path": str(source_path),
            }
        )
    return results


def load_huawei_sleep_intervals(sleepdata_dir: Path) -> pd.DataFrame:
    if not sleepdata_dir.exists():
        return pd.DataFrame(columns=["start_time", "end_time", "time_zone", "total_minutes", "source_path"])

    records: list[dict[str, Any]] = []
    for json_path in sorted(sleepdata_dir.rglob("*.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        for obj in _iter_json_objects(payload):
            if obj.get("key") != "SLEEP_RECORD":
                continue
            records.extend(_parse_sleep_interval_payload(obj, json_path))

    if not records:
        return pd.DataFrame(columns=["start_time", "end_time", "time_zone", "total_minutes", "source_path"])

    sleep_df = pd.DataFrame(records).sort_values(["start_time", "end_time"]).reset_index(drop=True)
    return sleep_df


def get_night_time_bounds(sensor_bundle: dict[str, pd.DataFrame]) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    frames = [frame for frame in sensor_bundle.values() if not frame.empty and "time" in frame.columns]
    if not frames:
        return None, None

    start = min(frame["time"].min() for frame in frames)
    end = max(frame["time"].max() for frame in frames)
    return start, end
