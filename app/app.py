from __future__ import annotations

import importlib
import time
import sys
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ml4b.config import DEFAULT_STEP_SECONDS, DEFAULT_WINDOW_SECONDS, EXAMPLE_NIGHTS_DIR, MODELS_DIR, SLEEPDATA_DIR
from ml4b.io import load_manual_labels
from ml4b.model import load_model_bundle
import ml4b.pipeline as pipeline
from ml4b.preprocess import NightFrameBundle, add_signal_magnitudes
from ml4b.labels import NightLabelSource, apply_binary_labels, resolve_label_source_for_night

pipeline = importlib.reload(pipeline)


st.set_page_config(page_title="ML4B Sleep Hypnogram", layout="wide")


MODE = st.sidebar.radio(
    "Model mode",
    ["Sleep phases", "Early version: Binary classification"],
    index=0,
)


def load_model_if_available(model_path: Path):
    if not model_path.exists():
        return None
    try:
        return load_model_bundle(model_path)
    except Exception as exc:
        st.warning(f"Could not load model artifact {model_path.name}: {exc}")
        return None


def discover_v1_model_artifacts(models_dir: Path) -> list[Path]:
    return sorted(models_dir.glob("sleep_model_w*_s*.joblib"), key=lambda path: path.name)


def parse_v1_model_spec(model_path: Path) -> tuple[int, int]:
    match = re.fullmatch(r"sleep_model_w(\d+)_s(\d+)\.joblib", model_path.name)
    if not match:
        raise ValueError(f"Unsupported V1 model name: {model_path.name}")
    return int(match.group(1)), int(match.group(2))


def downsample_for_plot(frame: pd.DataFrame, max_points: int = 4000) -> pd.DataFrame:
    if frame.empty or len(frame) <= max_points:
        return frame
    stride = max(1, len(frame) // max_points)
    return frame.iloc[::stride].copy()


def discover_example_nights(example_dir: Path) -> list[Path]:
    if not example_dir.exists():
        return []

    joblib_stems = {path.stem for path in example_dir.iterdir() if path.is_file() and path.suffix == ".joblib"}
    entries = [
        path
        for path in example_dir.iterdir()
        if path.name and (path.suffix == ".joblib" or (path.is_dir() and path.name not in joblib_stems))
    ]
    return sorted(entries, key=lambda path: (path.name, path.suffix != ".joblib"))


def format_example_night_label(path: Path) -> str:
    return path.stem if path.suffix == ".joblib" else path.name


def get_example_manual_labels_path(example_dir: Path) -> Path:
    return example_dir / "manual-labels.csv"


def apply_example_manual_labels(bundle: NightFrameBundle, manual_labels_path: Path) -> NightFrameBundle:
    if not manual_labels_path.exists() or bundle.frame.empty:
        return bundle

    manual_labels = load_manual_labels(manual_labels_path)
    if manual_labels.empty:
        return bundle

    label_night_id = bundle.night_id
    if label_night_id == "DEMO" and (manual_labels["night_id"] == "app-demo-night").any():
        label_night_id = "app-demo-night"

    night_start = bundle.frame["time"].min() if "time" in bundle.frame.columns else None
    night_end = bundle.frame["time"].max() if "time" in bundle.frame.columns else None
    label_source = resolve_label_source_for_night(
        label_night_id,
        night_start,
        night_end,
        manual_labels,
        pd.DataFrame(),
    )
    if not label_source.intervals:
        return bundle

    labeled_frame = apply_binary_labels(bundle.frame, label_source.intervals)
    return NightFrameBundle(
        night_id=bundle.night_id,
        frame=labeled_frame,
        metadata=bundle.metadata,
        label_source=label_source,
    )


@st.cache_data(show_spinner=False)
def load_example_night_bundle(example_entry: Path, sleepdata_dir: Path, manual_labels_path: Path) -> NightFrameBundle:
    if example_entry.suffix == ".joblib":
        payload = joblib.load(example_entry)
        if isinstance(payload, NightFrameBundle):
            return apply_example_manual_labels(payload, manual_labels_path)
        if hasattr(payload, "frame"):
            bundle = NightFrameBundle(
                night_id=getattr(payload, "night_id", example_entry.stem),
                frame=payload.frame,
                metadata=getattr(payload, "metadata", {}),
                label_source=getattr(payload, "label_source", NightLabelSource("example", [])),
            )
            return apply_example_manual_labels(bundle, manual_labels_path)
        if isinstance(payload, pd.DataFrame):
            bundle = NightFrameBundle(
                night_id=example_entry.stem,
                frame=payload,
                metadata={},
                label_source=NightLabelSource("example", []),
            )
            return apply_example_manual_labels(bundle, manual_labels_path)
        raise TypeError(f"Unsupported example night payload: {type(payload)!r}")

    return pipeline.load_or_build_night_frame(
        example_entry.name,
        raw_dir=example_entry.parent,
        sleepdata_dir=sleepdata_dir,
        manual_labels_path=manual_labels_path,
    )


@st.cache_data(show_spinner=False)
def load_sleep_phase_features_from_csv(csv_path: str) -> pd.DataFrame:
    records = []
    reader = pd.read_csv(
        csv_path,
        usecols=["time", "seconds_elapsed", "x", "y", "z"],
        chunksize=30 * 100,
        dtype={"time": "int64", "seconds_elapsed": "float32", "x": "float32", "y": "float32", "z": "float32"},
    )
    for i, chunk in enumerate(reader):
        mag = np.sqrt(chunk["x"].values ** 2 + chunk["y"].values ** 2 + chunk["z"].values ** 2)
        diffs = np.abs(np.diff(mag))
        records.append(
            {
                "epoch": i,
                "time_ns": int(chunk["time"].iloc[0]),
                "seconds": float(chunk["seconds_elapsed"].iloc[0]),
                "mean_mag": float(np.mean(mag)),
                "std_mag": float(np.std(mag)),
                "energy": float(np.mean(mag ** 2)),
                "activity_count": float(np.sum(diffs)),
                "iqr": float(np.percentile(mag, 75) - np.percentile(mag, 25)),
                "max_mag": float(np.max(mag)),
                "log_energy": float(np.log1p(np.mean(mag ** 2))),
            }
        )
    df = pd.DataFrame(records)
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["time_ns"], unit="ns", utc=True).dt.tz_convert("Europe/Berlin")
    return df


def load_sleep_phase_features(source: Path) -> pd.DataFrame:
    if source.suffix == ".joblib":
        payload = joblib.load(source)
        if hasattr(payload, "frame"):
            frame = payload.frame.copy()
        elif isinstance(payload, pd.DataFrame):
            frame = payload.copy()
        else:
            raise TypeError(f"Unsupported example night payload: {type(payload)!r}")

        # If the payload contains raw per-sample accelerometer readings but
        # not precomputed epoch features, aggregate into epoch-level
        # features (mean_mag, std_mag, energy, activity_count, ...).
        raw_accel_cols = {"accelerometer_x", "accelerometer_y", "accelerometer_z"}
        if "mean_mag" not in frame.columns and raw_accel_cols.issubset(set(frame.columns)):
            epoch_sec = DEFAULT_WINDOW_SECONDS if 'DEFAULT_WINDOW_SECONDS' in globals() else 30
            # compute per-sample magnitude
            mag = np.sqrt(frame["accelerometer_x"].values ** 2 + frame["accelerometer_y"].values ** 2 + frame["accelerometer_z"].values ** 2)
            # derive per-sample elapsed seconds if available, otherwise approximate
            if "seconds_elapsed" in frame.columns:
                seconds = frame["seconds_elapsed"].astype(float).values
            elif "time_ns" in frame.columns:
                t0 = pd.to_numeric(frame["time_ns"].iloc[0])
                seconds = (pd.to_numeric(frame["time_ns"]).values - t0) / 1e9
            else:
                # fallback: assume 100 Hz
                seconds = np.arange(len(frame)) / 100.0

            epoch_idx = (seconds // epoch_sec).astype(int)
            records = []
            for e, group_idx in enumerate(sorted(set(epoch_idx))):
                mask = epoch_idx == group_idx
                if not np.any(mask):
                    continue
                g_mag = mag[mask]
                # use first sample's time for epoch timestamp
                time_ns_vals = frame.loc[mask, "time_ns"] if "time_ns" in frame.columns else None
                records.append(
                    {
                        "epoch": int(e),
                        "time_ns": int(time_ns_vals.iloc[0]) if time_ns_vals is not None else None,
                        "seconds": float(seconds[mask][0]),
                        "mean_mag": float(np.mean(g_mag)),
                        "std_mag": float(np.std(g_mag)),
                        "energy": float(np.mean(g_mag ** 2)),
                        "activity_count": float(np.sum(np.abs(np.diff(g_mag)))),
                        "iqr": float(np.percentile(g_mag, 75) - np.percentile(g_mag, 25)),
                        "max_mag": float(np.max(g_mag)),
                        "log_energy": float(np.log1p(np.mean(g_mag ** 2))),
                    }
                )
            frame = pd.DataFrame(records)
            if not frame.empty and frame["time_ns"].notna().any():
                frame["datetime"] = pd.to_datetime(frame["time_ns"], unit="ns", utc=True).dt.tz_convert("Europe/Berlin")

        if "time_ns" not in frame.columns and "time" in frame.columns:
            frame = frame.copy()
            frame["time_ns"] = pd.to_numeric(frame["time"], errors="coerce")
        if "datetime" not in frame.columns and "time_ns" in frame.columns:
            frame["datetime"] = pd.to_datetime(frame["time_ns"], unit="ns", utc=True).dt.tz_convert("Europe/Berlin")
        return frame

    csv_path = source / "Accelerometer.csv"
    if not csv_path.exists():
        return pd.DataFrame()

    return load_sleep_phase_features_from_csv(str(csv_path))


def build_demo_probability_buckets(frame: pd.DataFrame, bucket_minutes: int = 10) -> pd.DataFrame:
    if frame.empty or "window_start" not in frame.columns or "sleep_probability" not in frame.columns:
        return pd.DataFrame()

    def mode_or_first(series: pd.Series):
        modes = series.mode(dropna=True)
        if not modes.empty:
            return modes.iloc[0]
        return series.dropna().iloc[0] if not series.dropna().empty else None

    bucketed = frame.copy()
    bucketed["bucket_start"] = pd.to_datetime(bucketed["window_start"]).dt.floor(f"{bucket_minutes}min")

    demo_frame = bucketed.groupby("bucket_start", as_index=False).agg(
        mean_sleep_probability=("sleep_probability", "mean"),
    )
    if "sample_count" in bucketed.columns:
        sample_counts = bucketed.groupby("bucket_start", as_index=False).agg(sample_count=("sample_count", "sum"))
        demo_frame = demo_frame.merge(sample_counts, on="bucket_start", how="left")
    if "predicted_label" in bucketed.columns:
        predicted_labels = bucketed.groupby("bucket_start", as_index=False).agg(predicted_label=("predicted_label", mode_or_first))
        demo_frame = demo_frame.merge(predicted_labels, on="bucket_start", how="left")
    if "label" in bucketed.columns:
        reference_labels = bucketed.groupby("bucket_start", as_index=False).agg(label=("label", mode_or_first))
        demo_frame = demo_frame.merge(reference_labels, on="bucket_start", how="left")

    demo_frame = demo_frame.rename(columns={"bucket_start": "window_start"})
    demo_frame["window_end"] = demo_frame["window_start"] + pd.Timedelta(minutes=bucket_minutes)
    return demo_frame.sort_values("window_start").reset_index(drop=True)


def _format_metric_value(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if pd.isna(value):
            return "n/a"
        return f"{value:.3f}"
    return str(value)


def render_metric_grid(metrics: dict[str, float]) -> None:
    validation_keys = ["accuracy", "balanced_accuracy", "f1", "precision", "recall", "roc_auc"]

    st.caption("Validation metrics come from grouped cross-validation across nights.")

    st.markdown("**Validation metrics**")
    validation_cols = st.columns(3)
    for index, key in enumerate(validation_keys):
        with validation_cols[index % 3]:
            st.metric(key.replace("_", " ").title(), _format_metric_value(metrics.get(key)))

def run_centered_visible_spinner(message: str, callback, min_seconds: float = 0.35):
    start_time = time.perf_counter()
    spinner_col = st.columns([1.25, 0.9, 1.25])[1]
    with spinner_col:
        with st.spinner(message):
            result = callback()
    elapsed = time.perf_counter() - start_time
    if elapsed < min_seconds:
        time.sleep(min_seconds - elapsed)
    return result


# =============================================================================
# V1: Binary classification
# =============================================================================
def run_binary_classification_mode() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at top left, rgba(57, 84, 255, 0.10), transparent 32%),
                        linear-gradient(180deg, #08111f 0%, #0f1726 48%, #111827 100%);
            color: #edf2ff;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1320px;
        }
        .hero-card {
            background: rgba(10, 18, 33, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 24px;
            padding: 1.5rem 1.75rem;
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
        }
        .section-card {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 20px;
            padding: 1rem 1.1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='hero-card'><h1 style='margin:0;'>ML4B Sleep Hypnogram</h1><p style='margin:0.35rem 0 0; opacity:0.8;'>Fuse raw phone sensors with smartwatch sleep intervals or manual fallback labels, train a window model, and visualize the predicted night.</br>Load time of widgets can take up to 60 seconds, please remain patient.</p></div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Workspace")
        sleepdata_dir = Path(st.text_input("Sleep export folder", value=str(SLEEPDATA_DIR)))
        example_dir = Path(st.text_input("Example nights folder", value=str(EXAMPLE_NIGHTS_DIR)))
        example_labels_path = get_example_manual_labels_path(example_dir)
        st.caption("Prefer compact .joblib demo bundles; raw demo folders are local-source only and are ignored in git.")
        model_options = discover_v1_model_artifacts(MODELS_DIR)
        if not model_options:
            st.warning("No V1 model artifacts were found in the models folder.")
            return
        model_path = st.selectbox("V1 model artifact", model_options, format_func=lambda path: path.name)
        window_seconds, step_seconds = parse_v1_model_spec(model_path)
        st.caption(f"Using {model_path.name} for {window_seconds}s windows and {step_seconds}s steps")

    example_night_entries = discover_example_nights(example_dir)
    night_options = [format_example_night_label(path) for path in example_night_entries]
    if not night_options:
        st.warning("No usable example night folders were found. Put a few sample nights into the example folder.")
        return

    selected_night_id = st.sidebar.selectbox("Example night", night_options, index=0)
    selected_example_entry = example_night_entries[night_options.index(selected_night_id)]
    def build_upper_widgets():
        night_bundle = load_example_night_bundle(
            selected_example_entry,
            sleepdata_dir=sleepdata_dir,
            manual_labels_path=example_labels_path,
        )
        night_frame = add_signal_magnitudes(night_bundle.frame)
        model_bundle = load_model_if_available(model_path)
        return night_bundle, night_frame, model_bundle

    try:
        night_bundle, night_frame, model_bundle = run_centered_visible_spinner(
            "Loading night, model summary, and validation metrics...",
            build_upper_widgets,
        )
    except Exception as exc:
        st.error(f"Failed to load example night data for {selected_night_id}: {exc}")
        return

    st.caption("This view shows the trained model's predictions on a demo night. Reference labels are loaded from example_nights/manual-labels.csv when available.")

    if model_bundle is None:
        st.error(f"Could not load the selected model artifact: {model_path.name}")
    left_col, right_col = st.columns([1.2, 0.8])

    with left_col:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader(f"Night {selected_night_id}")
        st.write(f"Label source: {night_bundle.label_source.source}")
        st.write(f"Samples: {len(night_frame):,}")
        st.write(f"Time span: {night_frame['time'].min()} to {night_frame['time'].max()}")
        display_frame = downsample_for_plot(night_frame[[column for column in ["time", "accelerometer_magnitude", "gyroscope_magnitude", "label"] if column in night_frame.columns]], max_points=1200)
        st.dataframe(display_frame.head(200), width='stretch', height=280)
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Model summary")
        if model_bundle is not None:
            meta_cols = st.columns(3)
            for column, key in zip(meta_cols, ["n_samples", "n_groups", "feature_count"]):
                with column:
                    st.metric(key.replace("_", " ").title(), model_bundle.metadata.get(key, 0))
            st.markdown("---")
            render_metric_grid(model_bundle.metrics)
        else:
            st.info("Train the model by keeping the default workspace in place, or create the model artifact with the CLI.")
        st.markdown("</div>", unsafe_allow_html=True)

    def build_lower_widgets():
        prediction_frame = pd.DataFrame()
        if model_bundle is not None:
            prediction_frame = pipeline.predict_hypnogram_for_night(
                night_frame,
                model_bundle,
                window_seconds=window_seconds,
                step_seconds=step_seconds,
            )

        demo_prediction_frame = build_demo_probability_buckets(prediction_frame, bucket_minutes=10)

        magnitude_figure = go.Figure()
        plot_frame = downsample_for_plot(night_frame, max_points=4000)
        if "accelerometer_magnitude" in plot_frame.columns:
            magnitude_figure.add_trace(go.Scatter(x=plot_frame["time"], y=plot_frame["accelerometer_magnitude"], name="Accelerometer magnitude", line=dict(color="#60a5fa", width=1.5)))
        if "gyroscope_magnitude" in plot_frame.columns:
            magnitude_figure.add_trace(go.Scatter(x=plot_frame["time"], y=plot_frame["gyroscope_magnitude"], name="Gyroscope magnitude", line=dict(color="#f59e0b", width=1.2, dash="dot")))
        magnitude_figure.update_layout(template="plotly_dark", height=360, margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation="h"))

        hypnogram_figure = go.Figure()
        plot_frame = demo_prediction_frame if not demo_prediction_frame.empty else prediction_frame
        if not plot_frame.empty:
            y_column = "mean_sleep_probability" if "mean_sleep_probability" in plot_frame.columns else "sleep_probability"
            hypnogram_figure.add_trace(
                go.Scatter(
                    x=plot_frame["window_start"],
                    y=plot_frame[y_column],
                    name="Sleep probability",
                    mode="lines+markers",
                    line=dict(color="#38bdf8", width=2),
                )
            )
            if "predicted_label" in plot_frame.columns:
                hypnogram_figure.add_trace(
                    go.Scatter(
                        x=plot_frame["window_start"],
                        y=(plot_frame["predicted_label"] == "SLEEP").astype(int),
                        name="Predicted label",
                        mode="lines",
                        line_shape="hv",
                        line=dict(color="#34d399", width=2),
                    )
                )
            if "label" in plot_frame.columns:
                hypnogram_figure.add_trace(
                    go.Scatter(
                        x=plot_frame["window_start"],
                        y=(plot_frame["label"] == "SLEEP").astype(int),
                        name="Reference label",
                        mode="lines",
                        line_shape="hv",
                        line=dict(color="#f97316", width=2, dash="dash"),
                    )
                )
        hypnogram_figure.update_yaxes(range=[-0.05, 1.05], tickvals=[0, 1], ticktext=["Awake", "Sleep"])
        hypnogram_figure.update_layout(template="plotly_dark", height=360, margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation="h"))

        raw_windows_frame = prediction_frame[[column for column in ["window_start", "window_end", "sleep_probability", "predicted_label", "label", "sleep_fraction", "sample_count"] if column in prediction_frame.columns]]
        return prediction_frame, demo_prediction_frame, magnitude_figure, hypnogram_figure, raw_windows_frame

    try:
        prediction_frame, demo_prediction_frame, magnitude_figure, hypnogram_figure, raw_windows_frame = run_centered_visible_spinner(
            "Loading remaining widgets...",
            build_lower_widgets,
        )
    except Exception as exc:
        st.error(f"Failed to build prediction widgets: {exc}")
        return

    chart_col_1, chart_col_2 = st.columns(2)

    with chart_col_1:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Sensor magnitude")
        st.plotly_chart(magnitude_figure, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

    with chart_col_2:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("10-minute demo hypnogram")
        st.plotly_chart(hypnogram_figure, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Raw model windows (debug)", expanded=False):
        if prediction_frame.empty:
            st.info("No prediction windows could be generated for this night.")
        else:
            st.dataframe(raw_windows_frame, width='stretch', height=320)

# =============================================================================
# Final: Sleep phases
# =============================================================================
def run_sleep_phase_mode() -> None:
    import pickle
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    EXAMPLE_DIR = EXAMPLE_NIGHTS_DIR
    EPOCH_SEC = 30
    SAMPLE_HZ = 100
    EPO_SAMPLES = EPOCH_SEC * SAMPLE_HZ

    STAGES = {0: "Tiefschlaf", 1: "Leichtschlaf", 2: "REM", 3: "Wach"}
    COLORS = {0: "#3B5BDB", 1: "#74C0FC", 2: "#9775FA", 3: "#FF6B6B"}
    FEATURE_COLS = ["mean_mag", "std_mag", "energy", "activity_count", "iqr", "max_mag", "log_energy"]

    def find_nights(example_dir: Path):
        """Liest alle Accelerometer-CSVs und zeigt echte Aufnahmedaten an."""
        nights = {}
        for night_entry in sorted([path for path in example_dir.iterdir() if path.is_dir() or path.suffix == ".joblib"], key=lambda path: path.name):
            # Datum aus dem Ordner- oder Dateinamen lesen z.B. 2026-05-11_22-17-20
            name = night_entry.stem if night_entry.suffix == ".joblib" else night_entry.name
            try:
                from datetime import datetime

                dt = datetime.strptime(name, "%Y-%m-%d_%H-%M-%S")
                label = dt.strftime("%d.%m.%Y  %H:%M Uhr")
            except ValueError:
                label = name
            nights[label] = night_entry
        return nights

    def heuristic_labels(df):
        # Ensure we have a sensible per-epoch activity signal. If the
        # precomputed `activity_count` is missing, try to derive a proxy
        # from available magnitude columns (`mean_mag` or
        # `accelerometer_magnitude`). Fall back to zeros.
        if "activity_count" in df.columns:
            ac = df["activity_count"].values
        elif "mean_mag" in df.columns:
            mag = df["mean_mag"].values
            diffs = np.abs(np.diff(mag, prepend=mag[0]))
            ac = diffs
        elif "accelerometer_magnitude" in df.columns:
            mag = df["accelerometer_magnitude"].values
            diffs = np.abs(np.diff(mag, prepend=mag[0]))
            ac = diffs
        else:
            ac = np.zeros(len(df))

        # Rank-normalise activity to get a simple percentile-based heuristic
        pct = ac.argsort().argsort() / max(1, len(df))
        labels = np.zeros(len(df), dtype=int)
        pos = np.linspace(0, 1, len(df))
        for i in range(len(df)):
            p = pct[i]
            if p > 0.85:
                labels[i] = 3
            elif p > 0.55:
                labels[i] = 1
            elif p > 0.20:
                labels[i] = 2 if pos[i] > 0.45 else 1
        return labels

    @st.cache_resource(show_spinner="Modell laden …")
    def get_model(paths: tuple):
        frames = []
        for p in paths:
            df = load_sleep_phase_features(Path(p))
            if df.empty:
                continue

            # Ensure feature columns exist. If missing, try to derive
            # reasonable fallbacks so the training step does not fail.
            def ensure_feature(col):
                if col in df.columns:
                    return
                if col == "activity_count":
                    if "mean_mag" in df.columns:
                        mag = df["mean_mag"].values
                        df["activity_count"] = np.abs(np.diff(mag, prepend=mag[0]))
                    elif "accelerometer_magnitude" in df.columns:
                        mag = df["accelerometer_magnitude"].values
                        df["activity_count"] = np.abs(np.diff(mag, prepend=mag[0]))
                    else:
                        df["activity_count"] = 0.0
                else:
                    # For other numeric features, fill with zeros if missing
                    df[col] = 0.0

            for col in FEATURE_COLS:
                ensure_feature(col)

            df["stage"] = heuristic_labels(df)
            frames.append(df)
        if not frames:
            raise RuntimeError("No training frames found when building the sleep-phase model. Check example night inputs.")
        combined = pd.concat(frames, ignore_index=True)
        X = combined[FEATURE_COLS].values
        y = combined["stage"].values
        scaler = StandardScaler()
        clf = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(scaler.fit_transform(X), y)
        return scaler, clf

    def predict_stages(df, scaler, clf):
        df = df.copy()
        df["stage"] = clf.predict(scaler.transform(df[FEATURE_COLS].values))
        df["stage_name"] = df["stage"].map(STAGES)
        return df

    def compute_metrics(df):
        def m(n):
            return n * EPOCH_SEC / 60

        total = len(df)
        wake = (df["stage"] == 3).sum()
        sleep = total - wake
        deep = (df["stage"] == 0).sum()
        rem = (df["stage"] == 2).sum()
        light = (df["stage"] == 1).sum()
        eff = 100 * m(sleep) / m(total) if total > 0 else 0
        return {
            "Schlafdauer": f"{int(m(sleep) // 60)}h {int(m(sleep) % 60):02d}min",
            "Schlafeffizienz": f"{eff:.0f} %",
            "Tiefschlaf": f"{m(deep):.0f} min",
            "REM-Schlaf": f"{m(rem):.0f} min",
            "Leichtschlaf": f"{m(light):.0f} min",
            "Wachphasen": f"{m(wake):.0f} min",
        }

    def plot_hypnogram(df):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=df["stage"],
                mode="lines",
                line=dict(color="rgba(130,130,150,0.4)", width=1.5, shape="hv"),
                showlegend=False,
            )
        )
        for s, c in COLORS.items():
            mask = df["stage"] == s
            fig.add_trace(
                go.Scatter(
                    x=df.loc[mask, "datetime"],
                    y=df.loc[mask, "stage"],
                    mode="markers",
                    marker=dict(color=c, size=4, symbol="square"),
                    name=STAGES[s],
                )
            )
        fig.update_layout(
            xaxis_title="Uhrzeit",
            yaxis=dict(tickvals=[0, 1, 2, 3], ticktext=["Tief", "Leicht", "REM", "Wach"], autorange="reversed", gridcolor="rgba(200,200,200,0.15)"),
            height=320,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="rgba(200,200,200,0.15)"),
        )
        return fig

    # Bissl UI-Design nh kann net schaden 😉
    st.title("🌙 Schlafanalyse Dashboard")
    st.caption("Sensordaten → Random Forest → Schlafphasen")

    example_dir = Path(st.sidebar.text_input("Example nights folder", value=str(EXAMPLE_DIR)))
    nights = find_nights(example_dir)
    if not nights:
        st.warning(f"Keine Aufnahmen gefunden in `{example_dir.resolve()}`")
        return

    selected_name = st.selectbox("Aufnahme auswählen", list(nights.keys()))
    selected_path = nights[selected_name]

    all_paths = tuple(str(p) for p in nights.values())
    scaler, clf = get_model(all_paths)

    df_raw = load_sleep_phase_features(selected_path)
    df = predict_stages(df_raw, scaler, clf)
    metrics = compute_metrics(df)

    icons = ["😴", "📊", "🌊", "💜", "💤", "👁️"]
    cols = st.columns(len(metrics))
    for col, (label, value), icon in zip(cols, metrics.items(), icons):
        col.metric(f"{icon} {label}", value)

    st.divider()
    st.subheader("Schlafverlauf der Nacht")
    st.plotly_chart(plot_hypnogram(df), width='stretch')
    st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Schlafphasen-Verteilung")
        dist = df["stage"].value_counts().rename(index=STAGES).reset_index()
        dist.columns = ["Phase", "Epochen"]
        dist["Minuten"] = (dist["Epochen"] * EPOCH_SEC / 60).round(1)
        fig_pie = px.pie(dist, values="Minuten", names="Phase", color="Phase", color_discrete_map={v: COLORS[k] for k, v in STAGES.items()}, hole=0.4)
        fig_pie.update_layout(margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h"))
        st.plotly_chart(fig_pie, width='stretch')

    with col_r:
        st.subheader("Feature-Wichtigkeit")
        imp = pd.DataFrame({"Feature": FEATURE_COLS, "Wichtigkeit": clf.feature_importances_}).sort_values("Wichtigkeit", ascending=True)
        fig_bar = px.bar(imp, x="Wichtigkeit", y="Feature", orientation="h", color="Wichtigkeit", color_continuous_scale=["#74C0FC", "#3B5BDB"])
        fig_bar.update_layout(margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False, xaxis=dict(gridcolor="rgba(200,200,200,0.15)"))
        st.plotly_chart(fig_bar, width='stretch')

    if len(nights) > 1:
        st.divider()
        st.subheader("Alle Aufnahmen im Vergleich")
        rows = []
        for name, path in nights.items():
            df_n = predict_stages(load_sleep_phase_features(path), scaler, clf)
            rows.append({"Aufnahme": name, **compute_metrics(df_n)})
        st.dataframe(pd.DataFrame(rows).set_index("Aufnahme"), width='stretch')


def main() -> None:
    try:
        if MODE == "Early version: Binary classification":
            run_binary_classification_mode()
        else:
            run_sleep_phase_mode()
    except Exception as exc:
        st.error("The app failed to start. Please check the repository assets and model files.")
        st.exception(exc)


main()
