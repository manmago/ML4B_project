from __future__ import annotations

import importlib
import pickle
import sys
from pathlib import Path

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

from ml4b.config import DEFAULT_STEP_SECONDS, DEFAULT_WINDOW_SECONDS, MODELS_DIR, RAW_DIR, SLEEPDATA_DIR
from ml4b.io import discover_night_dirs
from ml4b.model import load_model_bundle
import ml4b.pipeline as pipeline
from ml4b.preprocess import add_signal_magnitudes

pipeline = importlib.reload(pipeline)


st.set_page_config(page_title="ML4B Sleep Hypnogram", layout="wide")


MODE = st.sidebar.radio(
    "Model mode",
    ["V1: Binary classification", "Final: Sleep phases"],
    index=0,
)


def load_model_if_available(model_path: Path):
    if not model_path.exists():
        return None
    return load_model_bundle(model_path)


def downsample_for_plot(frame: pd.DataFrame, max_points: int = 4000) -> pd.DataFrame:
    if frame.empty or len(frame) <= max_points:
        return frame
    stride = max(1, len(frame) // max_points)
    return frame.iloc[::stride].copy()


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
    training_keys = ["train_accuracy", "train_balanced_accuracy", "train_f1", "train_precision", "train_recall", "train_roc_auc"]

    st.caption("Validation metrics come from grouped cross-validation across nights. Training metrics are shown for reference only.")

    st.markdown("**Validation metrics**")
    validation_cols = st.columns(3)
    for index, key in enumerate(validation_keys):
        with validation_cols[index % 3]:
            st.metric(key.replace("_", " ").title(), _format_metric_value(metrics.get(key)))

    st.markdown("**Training metrics**")
    training_cols = st.columns(3)
    for index, key in enumerate(training_keys):
        with training_cols[index % 3]:
            st.metric(key.replace("train_", "").replace("_", " ").title(), _format_metric_value(metrics.get(key)))


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
        raw_dir = Path(st.text_input("Raw data folder", value=str(RAW_DIR)))
        sleepdata_dir = Path(st.text_input("Sleep export folder", value=str(SLEEPDATA_DIR)))
        window_seconds = st.slider("Window size (seconds)", 10, 120, DEFAULT_WINDOW_SECONDS, 5)
        step_seconds = st.slider("Step size (seconds)", 5, 60, DEFAULT_STEP_SECONDS, 5)
        model_path = MODELS_DIR / f"sleep_model_w{window_seconds}_s{step_seconds}.joblib"
        st.caption(f"Model artifact: {model_path.name}")

    night_options = [path.name for path in discover_night_dirs(raw_dir)]
    if not night_options:
        st.error("No usable night folders were found in the raw data directory.")
        st.stop()

    selected_night_id = st.sidebar.selectbox("Night", night_options, index=0)

    night_status = st.sidebar.empty()
    night_status.info("loading.... night data")
    try:
        night_bundle = pipeline.load_or_build_night_frame(selected_night_id, raw_dir=raw_dir, sleepdata_dir=sleepdata_dir)
        night_frame = add_signal_magnitudes(night_bundle.frame)
        night_status.success(f"loaded night data: {selected_night_id}")
    except Exception as exc:
        night_status.error("failed to load night data")
        st.error(f"Failed to load night data for {selected_night_id}: {exc}")
        st.stop()

    model_status = st.sidebar.empty()
    model_status.info("loading.... model")
    model_bundle = None
    try:
        model_bundle = load_model_if_available(model_path)
    except Exception as exc:
        model_status.error("failed to load model")
        st.warning(f"Model could not be loaded or trained yet: {exc}")
    else:
        if model_bundle is None:
            model_status.error("failed to load model")
        else:
            model_status.success("loaded model")

    if model_bundle is None:
        st.warning(
            "No model artifact found for the selected window/step size. Run `python src/main.py train` after preprocessing or keep the default 30/15 settings with the existing model file."
        )

    left_col, right_col = st.columns([1.2, 0.8])

    with left_col:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader(f"Night {selected_night_id}")
        st.write(f"Label source: {night_bundle.label_source.source}")
        if night_bundle.label_source.source == "none":
            st.info("No matching health app or manual label interval was found for this night, so only prediction windows are shown.")
        st.write(f"Samples: {len(night_frame):,}")
        st.write(f"Time span: {night_frame['time'].min()} to {night_frame['time'].max()}")
        display_frame = downsample_for_plot(night_frame[[column for column in ["time", "accelerometer_magnitude", "gyroscope_magnitude", "label"] if column in night_frame.columns]], max_points=1200)
        st.dataframe(display_frame.head(200), use_container_width=True, height=280)
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

    prediction_frame = pd.DataFrame()
    if model_bundle is not None:
        prediction_frame = pipeline.predict_hypnogram_for_night(night_frame, model_bundle, window_seconds=window_seconds, step_seconds=step_seconds)

    demo_prediction_frame = build_demo_probability_buckets(prediction_frame, bucket_minutes=10)

    chart_col_1, chart_col_2 = st.columns(2)

    with chart_col_1:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Sensor magnitude")
        sensor_status = st.empty()
        sensor_status.info("loading.... sensor magnitude")
        magnitude_figure = go.Figure()
        try:
            plot_frame = downsample_for_plot(night_frame, max_points=4000)
            if "accelerometer_magnitude" in plot_frame.columns:
                magnitude_figure.add_trace(go.Scatter(x=plot_frame["time"], y=plot_frame["accelerometer_magnitude"], name="Accelerometer magnitude", line=dict(color="#60a5fa", width=1.5)))
            if "gyroscope_magnitude" in plot_frame.columns:
                magnitude_figure.add_trace(go.Scatter(x=plot_frame["time"], y=plot_frame["gyroscope_magnitude"], name="Gyroscope magnitude", line=dict(color="#f59e0b", width=1.2, dash="dot")))
            magnitude_figure.update_layout(template="plotly_dark", height=360, margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation="h"))
            st.plotly_chart(magnitude_figure, use_container_width=True)
            sensor_status.success("loaded sensor magnitude")
        except Exception as exc:
            sensor_status.error("failed to load sensor magnitude")
            st.error(f"Failed to load sensor magnitude: {exc}")
        st.markdown("</div>", unsafe_allow_html=True)

    with chart_col_2:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("10-minute demo hypnogram")
        hypnogram_status = st.empty()
        hypnogram_status.info("loading.... hypnogram")
        hypnogram_figure = go.Figure()
        try:
            if model_bundle is None:
                hypnogram_status.error("failed to load hypnogram")
                st.info("No model is loaded for the selected window/step size, so the hypnogram cannot be generated yet.")
            else:
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
                    hypnogram_status.success("loaded hypnogram")
                else:
                    hypnogram_status.error("failed to load hypnogram")
                    st.info("No prediction windows could be generated for this night.")
            hypnogram_figure.update_yaxes(range=[-0.05, 1.05], tickvals=[0, 1], ticktext=["Awake", "Sleep"])
            hypnogram_figure.update_layout(template="plotly_dark", height=360, margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation="h"))
            st.plotly_chart(hypnogram_figure, use_container_width=True)
        except Exception as exc:
            hypnogram_status.error("failed to load hypnogram")
            st.error(f"Failed to load hypnogram: {exc}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Raw model windows")
    window_status = st.empty()
    window_status.info("loading.... window predictions")
    try:
        if prediction_frame.empty:
            window_status.error("failed to load window predictions")
            st.info("No prediction windows could be generated for this night.")
        else:
            window_status.success("loaded window predictions")
            st.dataframe(
                prediction_frame[[column for column in ["window_start", "window_end", "sleep_probability", "predicted_label", "label", "sleep_fraction", "sample_count"] if column in prediction_frame.columns]],
                use_container_width=True,
                height=320,
            )
    except Exception as exc:
        window_status.error("failed to load window predictions")
        st.error(f"Failed to load window predictions: {exc}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("What the columns mean")
    st.markdown(
        """
        **10-minute demo hypnogram**
        - `window_start`: beginning of the 10-minute demo bucket shown in the chart.
        - `window_end`: end of the 10-minute demo bucket.
        - `mean_sleep_probability`: average sleep probability of all raw model windows inside that bucket. This is the value used in the demo chart.
        - `predicted_label`: the most common predicted state inside the bucket (`SLEEP` or `AWAKE`).
        - `label`: the most common reference label in that bucket, when a label source is available.
        - `sample_count`: number of raw prediction windows that were merged into the bucket.

        **Raw model windows**
        - `window_start`: start time of the exact model window.
        - `window_end`: end time of the exact model window.
        - `sleep_probability`: model confidence that the window is sleep. This directly drives the raw prediction output.
        - `predicted_label`: final class chosen from the model probability for that window.
        - `label`: reference label from Huawei/manual data, if available. It is used only for comparison and evaluation.
        - `sleep_fraction`: fraction of labeled samples inside the window that are marked as sleep.
        - `sample_count`: number of sensor samples in the model window.
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


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

    DATA_DIR = Path("data")
    MODEL_FILE = Path("sleep_model.pkl")
    EPOCH_SEC = 30
    SAMPLE_HZ = 100
    EPO_SAMPLES = EPOCH_SEC * SAMPLE_HZ

    STAGES = {0: "Tiefschlaf", 1: "Leichtschlaf", 2: "REM", 3: "Wach"}
    COLORS = {0: "#3B5BDB", 1: "#74C0FC", 2: "#9775FA", 3: "#FF6B6B"}
    FEATURE_COLS = ["mean_mag", "std_mag", "energy", "activity_count", "iqr", "max_mag", "log_energy"]

    def find_nights():
        """Liest alle Accelerometer-CSVs und zeigt echte Aufnahmedaten an."""
        nights = {}
        for p in sorted(DATA_DIR.glob("Accelerometer*.csv")):
            # Datum aus dem Dateinamen lesen z.B. 2026-05-11_22-17-20
            name = p.stem.replace("Accelerometer_", "")
            try:
                from datetime import datetime

                dt = datetime.strptime(name, "%Y-%m-%d_%H-%M-%S")
                label = dt.strftime("%d.%m.%Y  %H:%M Uhr")
            except ValueError:
                label = name
            nights[label] = p
        return nights

    @st.cache_data(show_spinner="Sensordaten verarbeiten …")
    def load_features(csv_path: str) -> pd.DataFrame:
        records = []
        reader = pd.read_csv(
            csv_path,
            usecols=["time", "seconds_elapsed", "x", "y", "z"],
            chunksize=EPO_SAMPLES,
            dtype={"time": "int64", "seconds_elapsed": "float32", "x": "float32", "y": "float32", "z": "float32"},
        )
        for i, chunk in enumerate(reader):
            if len(chunk) < EPO_SAMPLES // 2:
                break
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
        df["datetime"] = pd.to_datetime(df["time_ns"], unit="ns", utc=True).dt.tz_convert("Europe/Berlin")
        return df

    def heuristic_labels(df):
        ac = df["activity_count"].values
        pct = ac.argsort().argsort() / len(df)
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
        if MODEL_FILE.exists():
            return pickle.load(open(MODEL_FILE, "rb"))
        frames = []
        for p in paths:
            df = load_features(p)
            df["stage"] = heuristic_labels(df)
            frames.append(df)
        combined = pd.concat(frames, ignore_index=True)
        X = combined[FEATURE_COLS].values
        y = combined["stage"].values
        scaler = StandardScaler()
        clf = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(scaler.fit_transform(X), y)
        pickle.dump((scaler, clf), open(MODEL_FILE, "wb"))
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

    nights = find_nights()
    if not nights:
        st.error(f"Keine Aufnahmen gefunden in `{DATA_DIR.resolve()}`")
        st.stop()

    selected_name = st.selectbox("Aufnahme auswählen", list(nights.keys()))
    selected_path = nights[selected_name]

    all_paths = tuple(str(p) for p in nights.values())
    scaler, clf = get_model(all_paths)

    df_raw = load_features(str(selected_path))
    df = predict_stages(df_raw, scaler, clf)
    metrics = compute_metrics(df)

    icons = ["😴", "📊", "🌊", "💜", "💤", "👁️"]
    cols = st.columns(len(metrics))
    for col, (label, value), icon in zip(cols, metrics.items(), icons):
        col.metric(f"{icon} {label}", value)

    st.divider()
    st.subheader("Schlafverlauf der Nacht")
    st.plotly_chart(plot_hypnogram(df), use_container_width=True)
    st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Schlafphasen-Verteilung")
        dist = df["stage"].value_counts().rename(index=STAGES).reset_index()
        dist.columns = ["Phase", "Epochen"]
        dist["Minuten"] = (dist["Epochen"] * EPOCH_SEC / 60).round(1)
        fig_pie = px.pie(dist, values="Minuten", names="Phase", color="Phase", color_discrete_map={v: COLORS[k] for k, v in STAGES.items()}, hole=0.4)
        fig_pie.update_layout(margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h"))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_r:
        st.subheader("Feature-Wichtigkeit")
        imp = pd.DataFrame({"Feature": FEATURE_COLS, "Wichtigkeit": clf.feature_importances_}).sort_values("Wichtigkeit", ascending=True)
        fig_bar = px.bar(imp, x="Wichtigkeit", y="Feature", orientation="h", color="Wichtigkeit", color_continuous_scale=["#74C0FC", "#3B5BDB"])
        fig_bar.update_layout(margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False, xaxis=dict(gridcolor="rgba(200,200,200,0.15)"))
        st.plotly_chart(fig_bar, use_container_width=True)

    if len(nights) > 1:
        st.divider()
        st.subheader("Alle Aufnahmen im Vergleich")
        rows = []
        for name, path in nights.items():
            df_n = predict_stages(load_features(str(path)), scaler, clf)
            rows.append({"Aufnahme": name, **compute_metrics(df_n)})
        st.dataframe(pd.DataFrame(rows).set_index("Aufnahme"), use_container_width=True)


if MODE == "V1: Binary classification":
    run_binary_classification_mode()
else:
    run_sleep_phase_mode()
