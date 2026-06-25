from __future__ import annotations

import importlib
import shutil
import tempfile
import time
import sys
import re
import zipfile
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

from ml4b.config import DEFAULT_STEP_SECONDS, DEFAULT_WINDOW_SECONDS, EXAMPLE_NIGHTS_DIR, MODELS_DIR, SAMSUNG_SLEEP_PATH, SLEEPDATA_DIR, sleep_phase_model_filename
from ml4b.features import WindowSpec, feature_columns
from ml4b.io import load_manual_labels, load_night_sensor_bundle, load_sensor_csv
from ml4b.model import load_model_bundle
from ml4b.phases import load_phase_model_bundle, predict_phase_stages
import ml4b.pipeline as pipeline
from ml4b.preprocess import NightFrameBundle, add_signal_magnitudes, merge_sensor_streams
from ml4b.labels import NightLabelSource, apply_binary_labels, resolve_label_source_for_night
from ml4b.samsung import assign_window_stage_labels, load_samsung_sleep_intervals

pipeline = importlib.reload(pipeline)


st.set_page_config(page_title="ML4B Sleep Hypnogram", layout="wide")


MODE = st.sidebar.radio(
    "Model mode",
    ["Sleep phases", "Sleep phases (trained)", "Early version: Binary classification", "ℹ️ About"],
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
    # Matches both the original `sleep_model_w<win>_s<step>.joblib` and labelled variants such as
    # `sleep_model_samsung_w<win>_s<step>.joblib` (the accelerometer-only Samsung-trained model).
    # The canonical original (no variant token) is listed first so it stays the default selection.
    artifacts = [path for path in models_dir.glob("sleep_model_*.joblib") if _V1_MODEL_NAME_RE.fullmatch(path.name)]
    return sorted(artifacts, key=lambda path: (0 if _CANONICAL_V1_NAME_RE.fullmatch(path.name) else 1, path.name))


_V1_MODEL_NAME_RE = re.compile(r"sleep_model_(?:[a-z0-9]+_)?w(\d+)_s(\d+)\.joblib")
_CANONICAL_V1_NAME_RE = re.compile(r"sleep_model_w(\d+)_s(\d+)\.joblib")


def parse_v1_model_spec(model_path: Path) -> tuple[int, int]:
    match = _V1_MODEL_NAME_RE.fullmatch(model_path.name)
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


def extract_night_zip(uploaded_file, required: tuple[str, ...] = ("accelerometer.csv", "gyroscope.csv")) -> Path:
    """Extract Accelerometer.csv/Gyroscope.csv from an uploaded zip into a fresh temp directory.

    Searches the zip recursively (case-insensitively) for the required sensor files plus optional
    Metadata.csv/Annotation.csv, and extracts them flatly (ignoring any folder structure inside the
    zip) so the result matches the layout `ml4b.io.load_night_sensor_bundle` expects. ``required`` lists
    the lower-case filenames that must be present (the trained accel-only mode requires accelerometer only).
    """
    wanted = {
        "accelerometer.csv": "Accelerometer.csv",
        "gyroscope.csv": "Gyroscope.csv",
        "metadata.csv": "Metadata.csv",
        "annotation.csv": "Annotation.csv",
    }
    temp_dir = Path(tempfile.mkdtemp(prefix="ml4b_upload_"))
    try:
        with zipfile.ZipFile(uploaded_file) as archive:
            found: dict[str, str] = {}
            for member in archive.namelist():
                key = Path(member).name.lower()
                if key in wanted and key not in found:
                    found[key] = member
            for key, member in found.items():
                with archive.open(member) as source, open(temp_dir / wanted[key], "wb") as target:
                    shutil.copyfileobj(source, target)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    missing = [wanted[key] for key in required if key not in found]
    if missing:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise ValueError(f"Zip is missing required file(s): {', '.join(missing)}")
    return temp_dir


def build_uploaded_night_frame(night_dir: Path) -> pd.DataFrame:
    """Build a merged, magnitude-enriched night frame from an extracted Accelerometer/Gyroscope CSV pair."""
    sensor_bundle = load_night_sensor_bundle(night_dir)
    merged = merge_sensor_streams(
        sensor_bundle.get("accelerometer", pd.DataFrame()),
        sensor_bundle.get("gyroscope", pd.DataFrame()),
    )
    if merged.empty:
        raise ValueError("Could not merge accelerometer and gyroscope data from the uploaded night.")
    enriched = add_signal_magnitudes(merged)
    enriched["night_id"] = "uploaded"
    return enriched


def build_uploaded_accel_frame(uploaded_file) -> pd.DataFrame:
    """Build an accelerometer-only night frame from an uploaded Accelerometer CSV or a zip containing one.

    Used by the trained sleep-phase mode, whose model uses accelerometer features only — so gyroscope data
    is optional (ignored if present). Accepts either a raw `Accelerometer.csv` (the Sensor Logger / training
    format) or a `.zip` that contains one.
    """
    name = (uploaded_file.name or "").lower()
    if name.endswith(".zip"):
        temp_dir = extract_night_zip(uploaded_file, required=("accelerometer.csv",))
        try:
            sensor_bundle = load_night_sensor_bundle(temp_dir)
            accel = sensor_bundle.get("accelerometer", pd.DataFrame())
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    else:  # treat as a raw Accelerometer CSV
        temp_dir = Path(tempfile.mkdtemp(prefix="ml4b_accel_"))
        try:
            csv_path = temp_dir / "Accelerometer.csv"
            with open(csv_path, "wb") as target:
                target.write(uploaded_file.getbuffer())
            accel = load_sensor_csv(csv_path, "accelerometer")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    merged = merge_sensor_streams(accel, pd.DataFrame())
    if merged.empty:
        raise ValueError("Could not read any accelerometer samples from the upload.")
    enriched = add_signal_magnitudes(merged)
    enriched["night_id"] = "uploaded"
    return enriched


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


def render_night_analysis(
    night_frame: pd.DataFrame,
    model_bundle,
    window_seconds: int,
    step_seconds: int,
    key_prefix: str,
    spinner_message: str = "Building prediction widgets...",
    hypnogram_title: str = "10-minute hypnogram",
) -> pd.DataFrame:
    """Run inference on a night frame and render the magnitude/hypnogram charts plus a raw-windows table.

    Shared by the example-night view and the upload-your-own-night view so both produce identical charts.
    Returns the per-window prediction frame (empty if no model is loaded or no windows could be built).
    """

    def build():
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
            spinner_message,
            build,
        )
    except Exception as exc:
        st.error(f"Failed to build prediction widgets: {exc}")
        return pd.DataFrame()

    chart_col_1, chart_col_2 = st.columns(2)

    with chart_col_1:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Sensor magnitude")
        st.plotly_chart(magnitude_figure, width='stretch', key=f"{key_prefix}_magnitude_chart")
        st.markdown("</div>", unsafe_allow_html=True)

    with chart_col_2:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader(hypnogram_title)
        st.plotly_chart(hypnogram_figure, width='stretch', key=f"{key_prefix}_hypnogram_chart")
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Raw model windows (debug)", expanded=False):
        if prediction_frame.empty:
            st.info("No prediction windows could be generated for this night.")
        else:
            st.dataframe(raw_windows_frame, width='stretch', height=320, key=f"{key_prefix}_raw_windows")

    return prediction_frame


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
        def _model_label(path: Path) -> str:
            if "samsung" in path.name:
                return f"{path.name}  (accel-only · real Samsung labels)"
            return f"{path.name}  (accel + gyro · original)"

        model_path = st.selectbox("V1 model artifact", model_options, format_func=_model_label)
        window_seconds, step_seconds = parse_v1_model_spec(model_path)
        st.caption(f"Using {model_path.name} for {window_seconds}s windows and {step_seconds}s steps")
        if "samsung" in model_path.name:
            st.caption(
                "⚠️ This model is trained on real Samsung Health wake/sleep labels but uses the "
                "accelerometer only; its honest validation metrics are weak (balanced accuracy ≈ 0.60). "
                "See the model summary panel for details."
            )

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

    model_needs_gyro = model_bundle is not None and any("gyroscope" in column for column in model_bundle.feature_columns)
    night_has_gyro = any("gyroscope" in column for column in night_frame.columns)
    if model_needs_gyro and not night_has_gyro:
        st.info(
            f"`{selected_night_id}` is an **accelerometer-only** recording, but the selected model "
            f"`{model_path.name}` expects accelerometer **and** gyroscope features. Pick the accel+gyro demo "
            "night, or switch to the accelerometer-only Samsung model in the sidebar to analyze this night.",
            icon="ℹ️",
        )
    else:
        render_night_analysis(
            night_frame,
            model_bundle,
            window_seconds,
            step_seconds,
            key_prefix="example",
            spinner_message="Loading remaining widgets...",
            hypnogram_title="10-minute demo hypnogram",
        )

    st.divider()
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Analyze your own night")
    st.caption(
        "Upload a `.zip` containing `Accelerometer.csv` and `Gyroscope.csv` (e.g. exported from the "
        "Sensor Logger app, matching the layout under `data/raw/<night>/`). The file is processed in "
        "memory for this session only - nothing is stored on disk or sent anywhere else."
    )
    uploaded_file = st.file_uploader("Upload night recording (.zip)", type="zip", key="uploaded_night_zip")
    if uploaded_file is not None:
        if model_bundle is None:
            st.error("No model is loaded, cannot analyze the uploaded night.")
        else:
            temp_dir: Path | None = None
            try:
                temp_dir = extract_night_zip(uploaded_file)
                uploaded_frame = build_uploaded_night_frame(temp_dir)
                st.success(f"Loaded {len(uploaded_frame):,} merged samples from the uploaded night.")
                render_night_analysis(
                    uploaded_frame,
                    model_bundle,
                    window_seconds,
                    step_seconds,
                    key_prefix="uploaded",
                    spinner_message="Analyzing uploaded night...",
                    hypnogram_title="10-minute hypnogram",
                )
            except (ValueError, zipfile.BadZipFile) as exc:
                st.error(f"Could not process the uploaded file: {exc}")
            except Exception as exc:
                st.error(f"Failed to analyze the uploaded night: {exc}")
            finally:
                if temp_dir is not None:
                    shutil.rmtree(temp_dir, ignore_errors=True)
    st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# Final: Sleep phases
# =============================================================================
def run_sleep_phase_mode() -> None:
    from datetime import datetime

    EXAMPLE_DIR = EXAMPLE_NIGHTS_DIR
    WINDOW_SECONDS = DEFAULT_WINDOW_SECONDS
    STEP_SECONDS = DEFAULT_STEP_SECONDS

    STAGES = {0: "Deep sleep", 1: "Light sleep", 2: "REM", 3: "Awake"}
    COLORS = {0: "#3B5BDB", 1: "#74C0FC", 2: "#9775FA", 3: "#FF6B6B"}

    def find_nights(example_dir: Path):
        """Lists all available example nights (raw folders or .joblib bundles) with a readable label."""
        nights = {}
        for night_entry in sorted([path for path in example_dir.iterdir() if path.is_dir() or path.suffix == ".joblib"], key=lambda path: path.name):
            # Parse the date from the folder or file name, e.g. 2026-05-11_22-17-20
            name = night_entry.stem if night_entry.suffix == ".joblib" else night_entry.name
            try:
                dt = datetime.strptime(name, "%Y-%m-%d_%H-%M-%S")
                label = dt.strftime("%d.%m.%Y %H:%M")
            except ValueError:
                label = name
            nights[label] = night_entry
        return nights

    @st.cache_data(show_spinner=False)
    def load_night_features(example_entry: Path, sleepdata_dir: Path, manual_labels_path: Path) -> pd.DataFrame:
        """Build the same 64 window-statistics features (accelerometer + gyroscope) used by the binary model."""
        bundle = load_example_night_bundle(example_entry, sleepdata_dir=sleepdata_dir, manual_labels_path=manual_labels_path)
        night_frame = add_signal_magnitudes(bundle.frame)
        return pipeline.build_feature_dataset_for_prediction(
            [night_frame], window_spec=WindowSpec(window_seconds=WINDOW_SECONDS, step_seconds=STEP_SECONDS)
        )

    def heuristic_labels(df: pd.DataFrame) -> np.ndarray:
        """Rank windows by overall sensor activity (accel + gyro magnitude energy) and bucket them into
        deep/light/REM/awake stages with fixed percentile thresholds. This is only a heuristic used to
        create pseudo-labels for the exploratory phase classifier - see the disclosure banner below."""
        activity = df.get("accelerometer_magnitude_energy", pd.Series(np.zeros(len(df)))).to_numpy(dtype=float)
        if "gyroscope_magnitude_energy" in df.columns:
            activity = activity + df["gyroscope_magnitude_energy"].to_numpy(dtype=float)

        pct = activity.argsort().argsort() / max(1, len(df))
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

    @st.cache_resource(show_spinner="Loading model…")
    def get_model(paths: tuple, sleepdata_dir: Path, manual_labels_path: Path, feature_cols: tuple):
        frames = []
        for p in paths:
            df = load_night_features(Path(p), sleepdata_dir, manual_labels_path)
            if df.empty:
                continue
            # Skip nights that don't expose every selected feature column (e.g. accel-only nights
            # when the selected night is accel+gyro). Mixing them would inject NaNs and break the RF fit.
            if not set(feature_cols).issubset(df.columns):
                continue
            df = df.copy()
            df["stage"] = heuristic_labels(df)
            frames.append(df)
        if not frames:
            raise RuntimeError("No training frames found when building the sleep-phase model. Check example night inputs.")
        combined = pd.concat(frames, ignore_index=True)
        X = combined[list(feature_cols)].to_numpy(dtype=float)
        y = combined["stage"].to_numpy()
        scaler = StandardScaler()
        clf = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(scaler.fit_transform(X), y)
        return scaler, clf

    def predict_stages(df: pd.DataFrame, scaler, clf, feature_cols: tuple) -> pd.DataFrame:
        df = df.copy()
        df["stage"] = clf.predict(scaler.transform(df[list(feature_cols)].to_numpy(dtype=float)))
        df["stage_name"] = df["stage"].map(STAGES)
        return df

    def compute_metrics(df: pd.DataFrame) -> dict[str, str]:
        def m(n):
            return n * STEP_SECONDS / 60

        total = len(df)
        wake = (df["stage"] == 3).sum()
        sleep = total - wake
        deep = (df["stage"] == 0).sum()
        rem = (df["stage"] == 2).sum()
        light = (df["stage"] == 1).sum()
        eff = 100 * m(sleep) / m(total) if total > 0 else 0
        return {
            "Sleep duration": f"{int(m(sleep) // 60)}h {int(m(sleep) % 60):02d}min",
            "Sleep efficiency": f"{eff:.0f} %",
            "Deep sleep": f"{m(deep):.0f} min",
            "REM sleep": f"{m(rem):.0f} min",
            "Light sleep": f"{m(light):.0f} min",
            "Awake": f"{m(wake):.0f} min",
        }

    def plot_hypnogram(df: pd.DataFrame):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["window_start"],
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
                    x=df.loc[mask, "window_start"],
                    y=df.loc[mask, "stage"],
                    mode="markers",
                    marker=dict(color=c, size=4, symbol="square"),
                    name=STAGES[s],
                )
            )
        fig.update_layout(
            xaxis_title="Time",
            yaxis=dict(tickvals=[0, 1, 2, 3], ticktext=["Deep", "Light", "REM", "Awake"], autorange="reversed", gridcolor="rgba(200,200,200,0.15)"),
            height=320,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="rgba(200,200,200,0.15)"),
        )
        return fig

    st.title("🌙 Sleep Analysis Dashboard")
    st.caption("Sensor data → Random Forest → Sleep phases")
    st.info(
        "The 4-stage classifier (deep/light/REM/awake) shown here reuses the same engineered window "
        "statistics (per-axis accelerometer — and gyroscope when present — features over 120s windows / "
        "60s steps) as the binary model, but is trained on heuristic, activity-percentile-based "
        "pseudo-labels generated at runtime - it is **not** validated against ground-truth sleep-stage "
        "data. Treat its output as illustrative rather than clinically validated. The 'Early version: "
        "Binary classification' mode uses the actual trained-and-evaluated model from `models/`.",
        icon="ℹ️",
    )

    example_dir = Path(st.sidebar.text_input("Example nights folder", value=str(EXAMPLE_DIR)))
    nights = find_nights(example_dir)
    if not nights:
        st.warning(f"No recordings found in `{example_dir.resolve()}`")
        return

    selected_name = st.selectbox("Select recording", list(nights.keys()))
    selected_path = nights[selected_name]

    manual_labels_path = get_example_manual_labels_path(example_dir)
    all_paths = tuple(str(p) for p in nights.values())

    df_features = load_night_features(selected_path, SLEEPDATA_DIR, manual_labels_path)
    if df_features.empty:
        st.warning(f"Could not extract any window features for `{selected_name}`.")
        return

    feature_cols = tuple(feature_columns(df_features))
    scaler, clf = get_model(all_paths, SLEEPDATA_DIR, manual_labels_path, feature_cols)

    df = predict_stages(df_features, scaler, clf, feature_cols)
    metrics = compute_metrics(df)

    icons = ["😴", "📊", "🌊", "💜", "💤", "👁️"]
    cols = st.columns(len(metrics))
    for col, (label, value), icon in zip(cols, metrics.items(), icons):
        col.metric(f"{icon} {label}", value)

    st.divider()
    st.subheader("Sleep progression of the night")
    st.plotly_chart(plot_hypnogram(df), width='stretch')
    st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Sleep phase distribution")
        dist = df["stage"].value_counts().rename(index=STAGES).reset_index()
        dist.columns = ["Phase", "Windows"]
        dist["Minutes"] = (dist["Windows"] * STEP_SECONDS / 60).round(1)
        fig_pie = px.pie(dist, values="Minutes", names="Phase", color="Phase", color_discrete_map={v: COLORS[k] for k, v in STAGES.items()}, hole=0.4)
        fig_pie.update_layout(margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h"))
        st.plotly_chart(fig_pie, width='stretch')

    with col_r:
        st.subheader("Feature importance")
        imp = pd.DataFrame({"Feature": list(feature_cols), "Importance": clf.feature_importances_})
        imp = imp.sort_values("Importance", ascending=False).head(15).sort_values("Importance", ascending=True)
        fig_bar = px.bar(imp, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale=["#74C0FC", "#3B5BDB"])
        fig_bar.update_layout(margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False, xaxis=dict(gridcolor="rgba(200,200,200,0.15)"))
        st.plotly_chart(fig_bar, width='stretch')
        st.caption(f"Top 15 of {len(feature_cols)} engineered features by importance for the heuristic phase classifier.")

    if len(nights) > 1:
        st.divider()
        st.subheader("All recordings compared")
        rows = []
        skipped = []
        for name, path in nights.items():
            df_n_features = load_night_features(path, SLEEPDATA_DIR, manual_labels_path)
            if df_n_features.empty:
                continue
            # The model is trained on the selected recording's feature set. Recordings that don't
            # expose every selected feature column (e.g. an accelerometer-only night when the selected
            # night also has gyroscope) can't be predicted - skip them instead of raising a KeyError.
            if not set(feature_cols).issubset(df_n_features.columns):
                skipped.append(name)
                continue
            df_n = predict_stages(df_n_features, scaler, clf, feature_cols)
            rows.append({"Recording": name, **compute_metrics(df_n)})
        st.dataframe(pd.DataFrame(rows).set_index("Recording"), width='stretch')
        if skipped:
            st.caption(
                "Not comparable to the selected recording (missing sensor columns, e.g. gyroscope): "
                + ", ".join(skipped)
            )


# =============================================================================
# Sleep phases (trained on real Samsung Health labels)
# =============================================================================
def run_trained_phase_mode() -> None:
    from datetime import datetime

    STAGES = {0: "Deep sleep", 1: "Light sleep", 2: "REM", 3: "Awake"}
    COLORS = {0: "#3B5BDB", 1: "#74C0FC", 2: "#9775FA", 3: "#FF6B6B"}
    NAME_TO_CODE = {"Deep": 0, "Light": 1, "REM": 2, "Awake": 3}
    WINDOW_SECONDS = DEFAULT_WINDOW_SECONDS
    STEP_SECONDS = DEFAULT_STEP_SECONDS

    st.title("🌙 Sleep Analysis Dashboard — trained model")
    st.caption("Accelerometer → Random Forest (trained on Samsung Health stages) → Sleep phases")

    model_path = MODELS_DIR / sleep_phase_model_filename(WINDOW_SECONDS, STEP_SECONDS)
    if not model_path.exists():
        st.warning(
            f"No trained sleep-phase model found at `{model_path}`. "
            "Train it first with `python -m main train-phases` (needs the raw accelerometer nights locally)."
        )
        return
    bundle = load_phase_model_bundle(model_path)

    bal = bundle.metrics.get("balanced_accuracy")
    st.warning(
        "**Honest evaluation:** this 4-class model is trained on **real** Samsung Health sleep stages "
        "(not heuristic pseudo-labels), but it uses **accelerometer motion only**. Under leave-one-night-out "
        f"cross-validation it reaches a balanced accuracy of about **{bal:.2f}** (4-class chance = 0.25) and "
        "largely collapses to the majority 'Light sleep' class — phone motion alone cannot reliably separate "
        "REM/Deep/Light without heart-rate/HRV signals. Treat the hypnogram below as illustrative; the "
        "validation metrics and confusion matrix tell the real story.",
        icon="⚠️",
    )

    # Ground-truth overlay source (Samsung Health), loaded once and reused per night.
    samsung_intervals = load_samsung_sleep_intervals(SAMSUNG_SLEEP_PATH)

    def stage_minutes(frame: pd.DataFrame, code: int) -> float:
        return int((frame["stage"] == code).sum()) * STEP_SECONDS / 60

    def render_phase_night(feature_frame: pd.DataFrame, key_prefix: str, allow_overlay: bool) -> None:
        """Render the per-night widgets (summary cards, hypnogram + optional Samsung overlay, pie).

        Shared by the example-night view and the upload-your-own-night view so both render identically.
        ``allow_overlay`` enables the Samsung ground-truth overlay (only meaningful for nights the export
        covers — i.e. the bundled recordings, not arbitrary uploads).
        """
        predicted = predict_phase_stages(bundle, feature_frame)
        predicted["stage"] = predicted["stage_name"].map(NAME_TO_CODE)

        actual = assign_window_stage_labels(feature_frame, samsung_intervals) if (allow_overlay and not samsung_intervals.empty) else pd.DataFrame()
        has_actual = not actual.empty
        if has_actual:
            actual = actual.rename(columns={"stage": "actual_stage_name"})
            actual["actual_stage"] = actual["actual_stage_name"].map(NAME_TO_CODE)

        total_min = len(predicted) * STEP_SECONDS / 60
        sleep_min = total_min - stage_minutes(predicted, 3)
        metrics = {
            "Sleep duration": f"{int(sleep_min // 60)}h {int(sleep_min % 60):02d}min",
            "Sleep efficiency": f"{(100 * sleep_min / total_min) if total_min else 0:.0f} %",
            "Deep sleep": f"{stage_minutes(predicted, 0):.0f} min",
            "REM sleep": f"{stage_minutes(predicted, 2):.0f} min",
            "Light sleep": f"{stage_minutes(predicted, 1):.0f} min",
            "Awake": f"{stage_minutes(predicted, 3):.0f} min",
        }
        icons = ["😴", "📊", "🌊", "💜", "💤", "👁️"]
        cols = st.columns(len(metrics))
        for col, (label, value), icon in zip(cols, metrics.items(), icons):
            col.metric(f"{icon} {label}", value)

        st.subheader("Predicted sleep progression" + (" vs. Samsung ground truth" if has_actual else ""))
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=predicted["window_start"], y=predicted["stage"], mode="lines",
                line=dict(color="#38bdf8", width=1.8, shape="hv"), name="Predicted",
            )
        )
        if has_actual:
            fig.add_trace(
                go.Scatter(
                    x=actual["window_start"], y=actual["actual_stage"], mode="lines",
                    line=dict(color="#f97316", width=1.8, shape="hv", dash="dash"), name="Samsung (actual)",
                )
            )
        fig.update_layout(
            xaxis_title="Time",
            yaxis=dict(tickvals=[0, 1, 2, 3], ticktext=["Deep", "Light", "REM", "Awake"], autorange="reversed", gridcolor="rgba(200,200,200,0.15)"),
            height=340, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_hypnogram")
        if has_actual:
            st.caption(
                "⚠️ The bundled example nights include recordings the model was **trained on** (in-sample). On "
                "those, a close predicted-vs-actual match is optimistic — the leave-one-night-out metrics in "
                "'About the model' below are the honest measure. Demo bundles are also downsampled to stay "
                "lightweight, so the curve can differ slightly from the full-resolution prediction."
            )

        st.subheader("Predicted phase distribution")
        dist = predicted["stage"].value_counts().rename(index=STAGES).reset_index()
        dist.columns = ["Phase", "Windows"]
        dist["Minutes"] = (dist["Windows"] * STEP_SECONDS / 60).round(1)
        pie_col = st.columns([1, 1])[0]
        fig_pie = px.pie(dist, values="Minutes", names="Phase", color="Phase", color_discrete_map={v: COLORS[k] for k, v in STAGES.items()}, hole=0.4)
        fig_pie.update_layout(margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h"))
        pie_col.plotly_chart(fig_pie, width="stretch", key=f"{key_prefix}_pie")

    # --- Example night ---
    example_dir = Path(st.sidebar.text_input("Example nights folder", value=str(EXAMPLE_NIGHTS_DIR)))
    example_entries = discover_example_nights(example_dir)
    example_entries = [path for path in example_entries if path.suffix == ".joblib" or path.is_dir()]
    if not example_entries:
        st.warning(f"No example nights found in `{example_dir.resolve()}`.")
        return

    def night_label(path: Path) -> str:
        name = path.stem if path.suffix == ".joblib" else path.name
        try:
            return datetime.strptime(name, "%Y-%m-%d_%H-%M-%S").strftime("%d.%m.%Y %H:%M")
        except ValueError:
            return name

    labels = [night_label(path) for path in example_entries]
    selected = st.selectbox("Select recording", labels)
    selected_entry = example_entries[labels.index(selected)]

    manual_labels_path = get_example_manual_labels_path(example_dir)
    night_bundle = load_example_night_bundle(selected_entry, sleepdata_dir=SLEEPDATA_DIR, manual_labels_path=manual_labels_path)
    night_frame = add_signal_magnitudes(night_bundle.frame)

    feature_frame = pipeline.build_feature_dataset_for_prediction(
        [night_frame], window_spec=WindowSpec(window_seconds=WINDOW_SECONDS, step_seconds=STEP_SECONDS)
    )
    if feature_frame.empty:
        st.warning("Could not extract any window features for this night.")
        return

    render_phase_night(feature_frame, key_prefix="example", allow_overlay=True)

    # --- Analyze your own night ---
    st.divider()
    st.subheader("Analyze your own night")
    st.caption(
        "Upload your own recording to run the trained sleep-phase model on it. Accepts a raw "
        "`Accelerometer.csv` (the Sensor Logger / training format) **or** a `.zip` containing one — this "
        "model uses accelerometer motion only, so gyroscope data is optional. The file is processed in "
        "memory for this session only; nothing is stored or sent anywhere. No Samsung overlay is shown for "
        "uploads (the export only covers the bundled nights)."
    )
    uploaded_file = st.file_uploader("Upload Accelerometer.csv or a .zip", type=["csv", "zip"], key="uploaded_phase_file")
    if uploaded_file is not None:
        try:
            uploaded_frame = build_uploaded_accel_frame(uploaded_file)
            st.success(f"Loaded {len(uploaded_frame):,} accelerometer samples from the upload.")
            uploaded_features = pipeline.build_feature_dataset_for_prediction(
                [uploaded_frame], window_spec=WindowSpec(window_seconds=WINDOW_SECONDS, step_seconds=STEP_SECONDS)
            )
            if uploaded_features.empty:
                st.warning("Could not extract any window features from the upload (is it long enough?).")
            else:
                render_phase_night(uploaded_features, key_prefix="uploaded", allow_overlay=False)
        except (ValueError, zipfile.BadZipFile) as exc:
            st.error(f"Could not process the uploaded file: {exc}")
        except Exception as exc:
            st.error(f"Failed to analyze the uploaded night: {exc}")

    # --- About the model (describes the model itself, independent of the chosen night) ---
    st.divider()
    st.subheader("About the model")

    st.markdown("**Feature importance**")
    classifier = bundle.model.named_steps["classifier"]
    imp = pd.DataFrame({"Feature": bundle.feature_columns, "Importance": classifier.feature_importances_})
    imp = imp.sort_values("Importance", ascending=False).head(15).sort_values("Importance", ascending=True)
    imp_col = st.columns([1, 1])[0]
    fig_bar = px.bar(imp, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale=["#74C0FC", "#3B5BDB"])
    fig_bar.update_layout(margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False, xaxis=dict(gridcolor="rgba(200,200,200,0.15)"))
    imp_col.plotly_chart(fig_bar, width="stretch", key="about_feature_importance")
    st.caption(f"Top 15 of {len(bundle.feature_columns)} accelerometer features.")

    st.markdown("**Validation metrics (leave-one-night-out)**")
    mcols = st.columns(3)
    mcols[0].metric("Balanced accuracy", _format_metric_value(bundle.metrics.get("balanced_accuracy")))
    mcols[1].metric("Macro F1", _format_metric_value(bundle.metrics.get("macro_f1")))
    mcols[2].metric("Nights", bundle.metadata.get("n_groups", 0))

    per_class = pd.DataFrame(
        [
            {
                "Stage": stage,
                "Precision": bundle.metrics.get(f"{stage}_precision"),
                "Recall": bundle.metrics.get(f"{stage}_recall"),
                "F1": bundle.metrics.get(f"{stage}_f1"),
                "Train windows": bundle.metadata.get("class_counts", {}).get(stage),
            }
            for stage in bundle.classes
        ]
    )
    st.markdown("**Per-stage performance**")
    st.dataframe(per_class, width="stretch", hide_index=True)

    confusion = bundle.metadata.get("confusion_matrix")
    if confusion is not None:
        conf_labels = bundle.metadata.get("confusion_labels", bundle.classes)
        st.markdown("**Confusion matrix** (rows = true, columns = predicted)")
        conf_fig = px.imshow(
            confusion, x=conf_labels, y=conf_labels, text_auto=True, color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="True", color="Windows"),
        )
        conf_fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(conf_fig, width="stretch", key="about_confusion")


# =============================================================================
# About
# =============================================================================
def run_about() -> None:
    st.title("🌙 About this app")
    st.caption("What you're looking at, in plain language.")

    st.markdown(
        """
This app turns **phone-sensor motion** into a picture of a night's sleep. It records movement with a
phone accelerometer (and sometimes a gyroscope), chops the night into short overlapping windows, computes
simple statistics for each window, and feeds those to a **Random Forest** classifier. Pick a view with the
**Model mode** selector in the sidebar.
        """
    )

    st.subheader("The four views")
    st.markdown(
        """
- **Sleep phases** — A polished 4-stage dashboard (Deep / Light / REM / Awake). It's a *demo*: the model
  is trained on **made-up labels** derived from how much you moved, so it has **never been checked against
  real sleep data**. Treat it as eye-candy, not truth.
- **Sleep phases (trained)** — The **honest** 4-stage version, trained on **real Samsung Health sleep
  stages**. Its headline result: using phone motion alone, 4-stage classification scores about
  **0.24 balanced accuracy** (random guessing = 0.25). In other words, it barely beats a coin flip — phone
  movement can't separate REM/Deep/Light without heart-rate data. You can also **upload your own night** here.
- **Early version: Binary classification** — The original model that answers the *easier* question,
  **asleep vs. awake** (not which stage). This one genuinely works well (~**98% accuracy**). You can upload
  your own night here too.
- **ℹ️ About** — This page.
        """
    )

    st.subheader("How to read the charts")
    st.markdown(
        """
- **Summary cards** (😴 💤 🌊 …) — the night's totals: sleep duration, efficiency, and minutes per stage.
- **Hypnogram** — a stair-step line of the sleep stage over time (Awake at the top, Deep at the bottom). In
  the trained mode, the blue line is the model's guess and the orange dashed line is the **real Samsung
  ground truth**, so you can see exactly where they agree and disagree.
- **Phase distribution (donut)** — how the night split across the four stages.
- **Confusion matrix** (trained mode) — rows are the truth, columns are the guess. Almost everything piles
  into the "Light" column, which is the concrete proof that the 4-stage model collapses to one class.
        """
    )

    st.subheader("The example nights")
    st.markdown(
        """
- **DEMO** — a phone recording with accelerometer **+** gyroscope. Used by the binary model and the demo
  phase mode. It was **not** used to train any model.
- **14.05.2026 00:17** — a real **training night** (accelerometer only) with full Samsung ground truth,
  included so the trained mode can be cross-checked against genuine data. Because the model was trained on
  it, its per-night match looks better than reality (it's *in-sample*) — the validation metrics are the
  honest measure. (Its signal is downsampled so the demo stays lightweight.)
        """
    )

    st.success(
        "**One-sentence takeaway:** asleep-vs-awake from phone motion works great (~98%); full 4-stage "
        "sleep classification from phone motion alone basically doesn't (~chance) — and this app lets you "
        "see both the polished demo and the honest, ground-truth-checked reality side by side.",
        icon="✅",
    )


def main() -> None:
    try:
        if MODE == "Early version: Binary classification":
            run_binary_classification_mode()
        elif MODE == "Sleep phases (trained)":
            run_trained_phase_mode()
        elif MODE == "ℹ️ About":
            run_about()
        else:
            run_sleep_phase_mode()
    except Exception as exc:
        st.error("The app failed to start. Please check the repository assets and model files.")
        st.exception(exc)


main()
