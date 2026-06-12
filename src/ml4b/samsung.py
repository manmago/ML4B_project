"""Samsung Health sleep-stage export as a label source.

The Samsung Health `sleep_stage` CSV export pairs each motion recording with a
ground-truth 4-stage hypnogram (Awake / Light / Deep / REM). This module parses that
export into tidy stage intervals and assigns a stage to each feature window by
majority time-overlap, so the windows can be used as training labels for a multiclass
sleep-phase model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_TIMEZONE

# Samsung Health sleep_stage type codes.
STAGE_CODE_MAP = {
    "40001": "Awake",
    "40002": "Light",
    "40003": "Deep",
    "40004": "REM",
}

# The export has a one-line product header above the real column header, and every data
# row carries a trailing comma (one field more than there are column names), so we read
# it positionally with an explicit name list and drop the dangling `_extra` column.
_SAMSUNG_COLUMNS = [
    "create_sh_ver",
    "start_time",
    "sleep_id",
    "custom",
    "modify_sh_ver",
    "update_time",
    "create_time",
    "stage",
    "time_offset",
    "deviceuuid",
    "pkg_name",
    "end_time",
    "datauuid",
    "_extra",
]


def _parse_local_timestamps(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
    return parsed.dt.tz_localize(DEFAULT_TIMEZONE, ambiguous="NaT", nonexistent="shift_forward")


def load_samsung_sleep_intervals(path: Path) -> pd.DataFrame:
    """Load Samsung Health sleep-stage intervals as a `[start_time, end_time, stage]` frame.

    Timestamps are localized to `DEFAULT_TIMEZONE` (the export stores local wall-clock with a
    matching `time_offset`). Rows whose stage code is unknown or whose timestamps fail to parse
    are dropped. Returns an empty (typed) frame if the file is missing or unreadable.
    """
    columns = ["start_time", "end_time", "stage"]
    if not path.exists():
        return pd.DataFrame(columns=columns)

    raw = pd.read_csv(path, skiprows=2, header=None, names=_SAMSUNG_COLUMNS, dtype=str)
    if raw.empty:
        return pd.DataFrame(columns=columns)

    frame = pd.DataFrame(
        {
            "start_time": _parse_local_timestamps(raw["start_time"]),
            "end_time": _parse_local_timestamps(raw["end_time"]),
            "stage": raw["stage"].map(STAGE_CODE_MAP),
        }
    )
    frame = frame.dropna(subset=["start_time", "end_time", "stage"])
    frame = frame[frame["end_time"] > frame["start_time"]]
    return frame.sort_values("start_time").reset_index(drop=True)


def assign_window_stage_labels(feature_frame: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    """Stamp each feature window with the sleep stage it overlaps the most.

    For every window (`window_start`/`window_end`) the total overlap with each stage's intervals
    is accumulated and the stage with the largest overlap wins. Windows with no overlap at all
    are dropped, so the result contains only labeled windows (analogous to ``require_labels=True``).
    Adds a string ``stage`` column.
    """
    if feature_frame.empty or intervals.empty:
        return feature_frame.iloc[0:0].assign(stage=pd.Series(dtype=str))
    if "window_start" not in feature_frame.columns or "window_end" not in feature_frame.columns:
        return feature_frame.iloc[0:0].assign(stage=pd.Series(dtype=str))

    # Normalize every timestamp to int64 nanoseconds before comparing. Samsung times are parsed
    # at microsecond resolution while the window times are nanosecond resolution, so a naive
    # int64 cast would put the two on different scales and they would never overlap.
    def _to_ns(series: pd.Series) -> np.ndarray:
        return pd.to_datetime(series).dt.as_unit("ns").astype("int64").to_numpy()

    win_start = _to_ns(feature_frame["window_start"])
    win_end = _to_ns(feature_frame["window_end"])

    iv_start = _to_ns(intervals["start_time"])
    iv_end = _to_ns(intervals["end_time"])
    iv_stage = intervals["stage"].to_numpy()

    stage_names = sorted(set(iv_stage.tolist()))
    stage_index = {name: idx for idx, name in enumerate(stage_names)}
    iv_stage_idx = np.array([stage_index[name] for name in iv_stage])

    assigned: list[str | None] = []
    for w_start, w_end in zip(win_start, win_end):
        overlap = np.minimum(w_end, iv_end) - np.maximum(w_start, iv_start)
        overlap = np.clip(overlap, 0, None)
        if not overlap.any():
            assigned.append(None)
            continue
        totals = np.bincount(iv_stage_idx, weights=overlap, minlength=len(stage_names))
        assigned.append(stage_names[int(totals.argmax())])

    labeled = feature_frame.copy()
    labeled["stage"] = assigned
    return labeled.dropna(subset=["stage"]).reset_index(drop=True)
