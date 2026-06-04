import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import pickle

st.set_page_config(page_title="Schlafanalyse", page_icon="🌙", layout="wide")

DATA_DIR     = Path("data")
SAMSUNG_FILE = Path("samsung_sleep.csv")

EPOCH_SEC   = 30
SAMPLE_HZ   = 100
EPO_SAMPLES = EPOCH_SEC * SAMPLE_HZ

STAGES      = {0: "Tiefschlaf", 1: "Leichtschlaf", 2: "REM", 3: "Wach"}
COLORS      = {0: "#3B5BDB", 1: "#74C0FC", 2: "#9775FA", 3: "#FF6B6B"}
FEATURE_COLS = ["mean_mag","std_mag","energy","activity_count","iqr","max_mag","log_energy"]
SAMSUNG_MAP  = {40001: 3, 40002: 1, 40003: 0, 40004: 2}

#  Samsung Health Labels 
@st.cache_data
def load_samsung_labels():
    df = pd.read_csv(SAMSUNG_FILE, skiprows=1, index_col=False)
    df["start_time"]  = pd.to_datetime(df["start_time"])
    df["end_time"]    = pd.to_datetime(df["end_time"])
    df["stage_label"] = df["stage"].apply(lambda x: SAMSUNG_MAP.get(int(x), 1))
    return df[["start_time","end_time","stage_label"]]

def get_samsung_label(epoch_dt, samsung_df):
    t     = epoch_dt.replace(tzinfo=None)
    match = samsung_df[(samsung_df["start_time"] <= t) & (samsung_df["end_time"] > t)]
    return int(match.iloc[0]["stage_label"]) if len(match) > 0 else None

#  Nächte finden
def find_nights():
    nights = {}
    for p in sorted(DATA_DIR.glob("Accelerometer*.csv")):
        name = p.stem.replace("Accelerometer_", "")
        try:
            from datetime import datetime
            dt    = datetime.strptime(name, "%Y-%m-%d_%H-%M-%S")
            label = dt.strftime("%d.%m.%Y  %H:%M Uhr")
        except ValueError:
            label = name
        nights[label] = p
    return nights

#  Feature-Extraktion 
@st.cache_data(show_spinner="Sensordaten verarbeiten …")
def load_features(csv_path: str) -> pd.DataFrame:
    records = []
    reader  = pd.read_csv(csv_path,
        usecols=["time","seconds_elapsed","x","y","z"],
        chunksize=EPO_SAMPLES,
        dtype={"time":"int64","seconds_elapsed":"float32",
               "x":"float32","y":"float32","z":"float32"})
    for i, chunk in enumerate(reader):
        if len(chunk) < EPO_SAMPLES // 2:
            break
        mag   = np.sqrt(chunk["x"].values**2 + chunk["y"].values**2 + chunk["z"].values**2)
        diffs = np.abs(np.diff(mag))
        records.append({
            "epoch":          i,
            "time_ns":        int(chunk["time"].iloc[0]),
            "seconds":        float(chunk["seconds_elapsed"].iloc[0]),
            "mean_mag":       float(np.mean(mag)),
            "std_mag":        float(np.std(mag)),
            "energy":         float(np.mean(mag**2)),
            "activity_count": float(np.sum(diffs)),
            "iqr":            float(np.percentile(mag,75) - np.percentile(mag,25)),
            "max_mag":        float(np.max(mag)),
            "log_energy":     float(np.log1p(np.mean(mag**2))),
        })
    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["time_ns"], unit="ns", utc=True).dt.tz_convert("Europe/Berlin")
    return df

def assign_labels(df, samsung_df):
    labels = np.full(len(df), -1, dtype=int)
    for i, row in df.iterrows():
        lbl = get_samsung_label(row["datetime"], samsung_df)
        if lbl is not None:
            labels[i] = lbl
    valid = labels != -1
    return labels, valid

#  Leave-One-Out: Modell dynamisch je gewählter Testnacht
@st.cache_resource(show_spinner="Modell trainieren …")
def get_model(all_paths: tuple, test_path: str):
    """
    Leave-One-Out Cross-Validation:
    - test_path  = die im Dropdown gewählte Nacht (wird NICHT zum Training genutzt)
    - train_paths = alle anderen Nächte
    → kein Data Leakage egal welche Nacht ausgewählt wird
    """
    samsung_df  = load_samsung_labels()
    train_paths = [p for p in all_paths if p != test_path]

    # Training
    train_frames = []
    for p in train_paths:
        df = load_features(p)
        lbls, valid = assign_labels(df, samsung_df)
        df_v = df[valid].copy()
        df_v["stage"] = lbls[valid]
        if len(df_v) > 0:
            train_frames.append(df_v)

    train_df = pd.concat(train_frames, ignore_index=True)
    X_train  = train_df[FEATURE_COLS].values
    y_train  = train_df["stage"].values

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=10,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    clf.fit(X_tr_sc, y_train)

    # Test auf der gewählten Nacht
    df_test = load_features(test_path)
    lbls_t, valid_t = assign_labels(df_test, samsung_df)
    df_test_v = df_test[valid_t].copy()
    df_test_v["stage"] = lbls_t[valid_t]

    if len(df_test_v) > 0:
        X_te_sc = scaler.transform(df_test_v[FEATURE_COLS].values)
        y_pred  = clf.predict(X_te_sc)
        y_true  = df_test_v["stage"].values
        acc     = float(np.mean(y_pred == y_true))
        cm      = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])
    else:
        acc = 0.0
        cm  = np.zeros((4,4), dtype=int)

    return scaler, clf, acc, cm

#  Vorhersage & Kennzahlen
def predict_stages(df, scaler, clf):
    df = df.copy()
    df["stage"]      = clf.predict(scaler.transform(df[FEATURE_COLS].values))
    df["stage_name"] = df["stage"].map(STAGES)
    return df

def compute_metrics(df):
    def m(n): return n * EPOCH_SEC / 60
    total = len(df)
    wake  = (df["stage"] == 3).sum()
    sleep = total - wake
    deep  = (df["stage"] == 0).sum()
    rem   = (df["stage"] == 2).sum()
    light = (df["stage"] == 1).sum()
    eff   = 100 * m(sleep) / m(total) if total > 0 else 0
    return {
        "Schlafdauer":     f"{int(m(sleep)//60)}h {int(m(sleep)%60):02d}min",
        "Schlafeffizienz": f"{eff:.0f} %",
        "Tiefschlaf":      f"{m(deep):.0f} min",
        "REM-Schlaf":      f"{m(rem):.0f} min",
        "Leichtschlaf":    f"{m(light):.0f} min",
        "Wachphasen":      f"{m(wake):.0f} min",
    }

def plot_hypnogram(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df["stage"], mode="lines",
        line=dict(color="rgba(130,130,150,0.4)", width=1.5, shape="hv"),
        showlegend=False))
    for s, c in COLORS.items():
        mask = df["stage"] == s
        fig.add_trace(go.Scatter(
            x=df.loc[mask,"datetime"], y=df.loc[mask,"stage"],
            mode="markers", marker=dict(color=c, size=4, symbol="square"),
            name=STAGES[s]))
    fig.update_layout(
        xaxis_title="Uhrzeit",
        yaxis=dict(tickvals=[0,1,2,3], ticktext=["Tief","Leicht","REM","Wach"],
                   autorange="reversed", gridcolor="rgba(200,200,200,0.15)"),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10,r=10,t=10,b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(200,200,200,0.15)"))
    return fig

def plot_confusion_matrix(cm):
    labels   = [STAGES[i] for i in range(4)]
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct   = np.divide(cm.astype(float), row_sums,
                         where=row_sums!=0, out=np.zeros_like(cm, dtype=float)) * 100

    # Helle Farbskala 
    fig = go.Figure(go.Heatmap(
        z=cm_pct,
        x=labels, y=labels,
        zmin=0, zmax=100,
        colorscale=[
            [0.0,  "rgba(240,244,255,1)"],
            [0.3,  "rgba(155,185,240,1)"],
            [0.6,  "rgba(80,130,210,1)"],
            [1.0,  "rgba(30,70,170,1)"],
        ],
        showscale=False,
        text=[[f"{int(cm[i,j])}<br>({cm_pct[i,j]:.0f}%)"
               for j in range(4)] for i in range(4)],
        texttemplate="%{text}",
        textfont=dict(size=13, color="black"),
    ))
    fig.update_layout(
        xaxis_title="Vorhergesagt",
        yaxis_title="Tatsächlich (Samsung)",
        yaxis=dict(autorange="reversed"),
        height=340,
        margin=dict(l=10,r=10,t=10,b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

#  UI 
st.title("🌙 Schlafanalyse Dashboard")
st.caption("Sensordaten → Random Forest → Schlafphasen")

nights = find_nights()
if not nights:
    st.error(f"Keine Aufnahmen in `{DATA_DIR.resolve()}`")
    st.stop()

selected_name = st.selectbox("Aufnahme auswählen", list(nights.keys()))
selected_path = str(nights[selected_name])
all_paths     = tuple(str(p) for p in nights.values())

# Leave-One-Out: gewählte Nacht = Testnacht, Rest = Training
scaler, clf, acc, cm = get_model(all_paths, selected_path)

df_raw  = load_features(selected_path)
df      = predict_stages(df_raw, scaler, clf)
metrics = compute_metrics(df)

icons = ["😴","📊","🌊","💜","💤","👁️"]
cols  = st.columns(len(metrics))
for col,(label,value),icon in zip(cols,metrics.items(),icons):
    col.metric(f"{icon} {label}", value)

st.divider()
st.subheader("Schlafverlauf der Nacht")
st.plotly_chart(plot_hypnogram(df), use_container_width=True)
st.divider()

col_l, col_r = st.columns(2)
with col_l:
    st.subheader("Schlafphasen-Verteilung")
    dist = df["stage"].value_counts().rename(index=STAGES).reset_index()
    dist.columns = ["Phase","Epochen"]
    dist["Minuten"] = (dist["Epochen"] * EPOCH_SEC / 60).round(1)
    fig_pie = px.pie(dist, values="Minuten", names="Phase",
        color="Phase",
        color_discrete_map={v: COLORS[k] for k,v in STAGES.items()},
        hole=0.4)
    fig_pie.update_layout(margin=dict(l=0,r=0,t=10,b=0),
        paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h"))
    st.plotly_chart(fig_pie, use_container_width=True)

with col_r:
    st.subheader("Feature-Wichtigkeit")
    imp = pd.DataFrame({"Feature": FEATURE_COLS,
                        "Wichtigkeit": clf.feature_importances_})\
            .sort_values("Wichtigkeit", ascending=True)
    fig_bar = px.bar(imp, x="Wichtigkeit", y="Feature", orientation="h",
        color="Wichtigkeit", color_continuous_scale=["#74C0FC","#3B5BDB"])
    fig_bar.update_layout(margin=dict(l=0,r=0,t=10,b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        xaxis=dict(gridcolor="rgba(200,200,200,0.15)"))
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()
col_cm, col_acc = st.columns([2,1])
with col_cm:
    st.subheader("Konfusionsmatrix")
    st.caption(f"Trainiert auf {len(nights)-1} Nächten — getestet auf: {selected_name}")
    st.plotly_chart(plot_confusion_matrix(cm), use_container_width=True)

with col_acc:
    st.subheader("Modell-Genauigkeit")
    st.metric("Accuracy (Test)", f"{acc*100:.1f} %")
    st.caption(
        "Die gewählte Nacht wurde beim Training "
        "komplett ausgeschlossen (Leave-One-Out). "
        "Das Modell hat diese Daten nie gesehen."
    )

if len(nights) > 1:
    st.divider()
    st.subheader("Alle Aufnahmen im Vergleich")
    rows = []
    for name, path in nights.items():
        df_n = predict_stages(load_features(str(path)), scaler, clf)
        rows.append({"Aufnahme": name, **compute_metrics(df_n)})
    st.dataframe(pd.DataFrame(rows).set_index("Aufnahme"), use_container_width=True)
