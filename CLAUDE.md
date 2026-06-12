# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

ML4B is a sleep-classification project: phone accelerometer/gyroscope data (recorded with the Sensor Logger app) is windowed into features and fed to a Random Forest to classify sleep vs. wake (and, in the demo app, light/deep/REM phases). A Streamlit app visualizes predictions as a hypnogram. See `project.md` for the full write-up (motivation, methodology, limitations) and `README.md` for the user-facing tutorial.

## Setup & commands

- Package manager is **uv** (see `SETUP.md`). Install deps: `uv sync`, activate with `source .venv/bin/activate`.
- Python `>=3.11` (see `.python-version` and `pyproject.toml`).
- Run the Streamlit app: `streamlit run app/app.py` (uses `src/ml4b` via a `sys.path` shim, so it works without installing the package).
- CLI entry point is `src/main.py`, invoked as a module (`python -m main <command>` from `src/`, or via an installed console script). Subcommands:
  - `build` — build the labeled dataset + feature table, write to CSV (`--output`, default `data/processed/feature_dataset.csv`)
  - `preprocess` — build/cache labeled night frames and feature windows (`--force-rebuild` to bypass caches)
  - `train` — train and save a `SleepModelBundle` (default model path `models/sleep_model_w{window}_s{step}.joblib`)
  - `predict` — load a model and print a hypnogram preview for one night (`--night-id`, defaults to the first discovered night)
  - All four accept `--raw-dir`, `--sleepdata-dir`, `--window-seconds`, `--step-seconds` (defaults: 120s window / 60s step, see `ml4b.config`)
- There is no test suite, linter, or CI configured in this repo.

## Architecture

### Data layout (see `ml4b.config` for all paths)
- `data/raw/<night_id>/` — one folder per recording, named `YYYY-MM-DD_HH-MM-SS` (matched by `NIGHT_FOLDER_PATTERN` in `ml4b.io`). Each contains `Accelerometer.csv`, `Gyroscope.csv`, `Metadata.csv`, `Annotation.csv`, etc. Ignored by git.
- `data/raw/manual-labels.csv` — manual `night_id,bed_time,wake_time` fallback labels.
- `data/sleepdata/` — Huawei Health export JSON files containing `SLEEP_RECORD` entries (primary label source when available).
- `data/processed/nights/<night_id>.joblib` — cached `NightFrameBundle` per night (merged sensor frame + labels).
- `data/processed/features/feature_dataset_w{window}_s{step}.joblib` — cached feature table for a given window/step config.
- `data/example_nights/` — small, git-tracked demo bundles (`DEMO.joblib` + `manual-labels.csv`) used by the Streamlit app so it can run without raw data.
- `models/sleep_model_w120_s60.joblib` — the only model artifact committed to git (`.gitignore` excludes other `models/*.joblib`).

### Pipeline (`src/ml4b/`)
The flow is: **io → labels → preprocess → features → model → pipeline** (orchestration).

1. **`io.py`** — loads raw sensor CSVs (converts `time` from Unix nanoseconds to tz-aware `Europe/Berlin` datetimes, prefixes axis columns with the sensor name, e.g. `accelerometer_x`), loads manual labels, and parses Huawei Health JSON exports for sleep intervals (`load_huawei_sleep_intervals`).
2. **`labels.py`** — `resolve_label_source_for_night` picks the label source per night: Huawei smartwatch intervals take priority over manual annotations, falling back to `"none"` (unlabeled). `apply_binary_labels` stamps each sample as `AWAKE`/`SLEEP`/`UNKNOWN` based on the resolved intervals.
3. **`preprocess.py`** — `merge_sensor_streams` aligns accelerometer and gyroscope via `pd.merge_asof` (nearest, `DEFAULT_MERGE_TOLERANCE_MS` tolerance), `prepare_night_frame`/`build_or_load_night_frame` produce a `NightFrameBundle` (night_id, merged+labeled frame, metadata, label source), with joblib caching to `PREPROCESSED_NIGHTS_DIR`. `add_signal_magnitudes` derives `accelerometer_magnitude`/`gyroscope_magnitude`.
4. **`features.py`** — `extract_window_features_with_labels` slides fixed-size, fixed-step windows (in nanoseconds) over a night frame and computes per-signal stats (mean/std/min/max/median/iqr/energy/range) for accelerometer/gyroscope axes + magnitudes. `require_labels=True` (training) drops unlabeled windows; `require_labels=False` (prediction) keeps them. `build_feature_dataset` / `build_feature_dataset_for_prediction` concatenate across nights.
5. **`model.py`** — `SleepModelBundle` wraps a sklearn `Pipeline` (median imputer + `RandomForestClassifier`, `class_weight="balanced_subsample"`). `train_sleep_model` excludes non-feature columns (`night_id`, `window_start`, `window_end`, `label`, `sleep_fraction`, `sample_count`), evaluates with `GroupKFold` grouped by `night_id` (so a night never spans train/test), and also reports in-sample train metrics. `predict_sleep_probability` outputs `sleep_probability` + `predicted_label`.
6. **`pipeline.py`** — top-level orchestration: `build_workspace_dataset` (with feature-cache short-circuit), `train_workspace_model`, `preprocess_workspace`, `predict_hypnogram_for_night`, `load_or_build_night_frame`.

### Streamlit app (`app/app.py`)
Single-file app with a sidebar radio toggling between two independent modes (`main()` dispatches):
- **"Sleep phases"** (`run_sleep_phase_mode`) — the primary/default mode. Loads example nights (raw folders or `.joblib` bundles) via `load_night_features`, which reuses the same 64 ml4b window features as the binary model (`build_feature_dataset_for_prediction`, 120s windows / 60s steps) rather than computing its own features. It then trains an in-memory `StandardScaler` + `RandomForestClassifier` against **heuristic** activity-percentile labels (`heuristic_labels`, derived at runtime from `accelerometer_magnitude_energy` + `gyroscope_magnitude_energy`) to assign 4-class stages (Deep sleep/Light sleep/REM/Awake). This is a self-contained demo trained on pseudo-labels — it reuses `ml4b`'s feature pipeline but not its trained binary model or real labels.
- **"Early version: Binary classification"** (`run_binary_classification_mode`) — uses the real `ml4b` pipeline and a saved V1 model artifact (`models/sleep_model_w*_s*.joblib`, parsed via `parse_v1_model_spec`) to predict AWAKE/SLEEP per window and compare against reference labels from `data/example_nights/manual-labels.csv`.

Both modes read from `data/example_nights/` by default (configurable via sidebar text input) and use `discover_example_nights` to support both `.joblib` bundles and raw night folders. `app/app.py` adds `ROOT` and `ROOT/src` to `sys.path` at import time so `ml4b` is importable without installation — `bin/app.py` is an older/alternate standalone variant of the sleep-phase demo (not using `ml4b` at all) and is largely superseded by `app/app.py`.

## Key conventions
- All timestamps are converted to `Europe/Berlin` (`DEFAULT_TIMEZONE` in `ml4b.config`); raw sensor times are Unix nanoseconds (`time_ns`).
- Window/step sizes are encoded into cache and model filenames (`feature_dataset_w{window}_s{step}.joblib`, `sleep_model_w{window}_s{step}.joblib`) — keep this naming when adding new artifacts.
- Demo data in `data/example_nights/` must stay small (compact `.joblib`, 3x-compressed) due to GitHub's 100MB limit; raw demo folders are for local regeneration only and are gitignored.
