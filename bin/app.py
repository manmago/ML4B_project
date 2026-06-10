import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Schlafanalyse", page_icon="🌙", layout="wide")

DATA_DIR    = Path("data")
EPOCH_SEC   = 30
SAMPLE_HZ   = 100
EPO_SAMPLES = EPOCH_SEC * SAMPLE_HZ

STAGES = {0: "Tiefschlaf", 1: "Leichtschlaf", 2: "REM", 3: "Wach"}
COLORS = {0: "#3B5BDB", 1: "#74C0FC", 2: "#9775FA", 3: "#FF6B6B"}
FEATURE_COLS = ["mean_mag","std_mag","energy","activity_count","iqr","max_mag","log_energy"]

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
    reader  = pd.read_csv(csv_path,
        usecols=["time","seconds_elapsed","x","y","z"], chunksize=EPO_SAMPLES,
        dtype={"time":"int64","seconds_elapsed":"float32",
               "x":"float32","y":"float32","z":"float32"})
    for i, chunk in enumerate(reader):
        mag   = np.sqrt(chunk["x"].values**2+chunk["y"].values**2+chunk["z"].values**2)
        diffs = np.abs(np.diff(mag))
        records.append({
            "epoch": i, "time_ns": int(chunk["time"].iloc[0]),
            "seconds": float(chunk["seconds_elapsed"].iloc[0]),
            "mean_mag": float(np.mean(mag)), "std_mag": float(np.std(mag)),
            "energy": float(np.mean(mag**2)), "activity_count": float(np.sum(diffs)),
            "iqr": float(np.percentile(mag,75)-np.percentile(mag,25)),
            "max_mag": float(np.max(mag)), "log_energy": float(np.log1p(np.mean(mag**2))),
        })
    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["time_ns"],unit="ns",utc=True).dt.tz_convert("Europe/Berlin")
    return df

def heuristic_labels(df):
    ac  = df["activity_count"].values
    pct = ac.argsort().argsort() / len(df)
    labels = np.zeros(len(df), dtype=int)
    pos    = np.linspace(0,1,len(df))
    for i in range(len(df)):
        p = pct[i]
        if   p > 0.85: labels[i] = 3
        elif p > 0.55: labels[i] = 1
        elif p > 0.20: labels[i] = 2 if pos[i] > 0.45 else 1
    return labels

@st.cache_resource(show_spinner="Modell laden …")
def get_model(paths: tuple):
    frames = []
    for p in paths:
        df = load_features(p)
        df["stage"] = heuristic_labels(df)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    X = combined[FEATURE_COLS].values
    y = combined["stage"].values
    scaler = StandardScaler()
    clf    = RandomForestClassifier(n_estimators=300,max_depth=10,
                class_weight="balanced",random_state=42,n_jobs=-1)
    clf.fit(scaler.fit_transform(X), y)
    return scaler, clf

def predict_stages(df, scaler, clf):
    df = df.copy()
    df["stage"]      = clf.predict(scaler.transform(df[FEATURE_COLS].values))
    df["stage_name"] = df["stage"].map(STAGES)
    return df

def compute_metrics(df):
    def m(n): return n*EPOCH_SEC/60
    total=len(df); wake=(df["stage"]==3).sum(); sleep=total-wake
    deep=(df["stage"]==0).sum(); rem=(df["stage"]==2).sum(); light=(df["stage"]==1).sum()
    eff=100*m(sleep)/m(total) if total>0 else 0
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
    fig.add_trace(go.Scatter(x=df["datetime"],y=df["stage"],mode="lines",
        line=dict(color="rgba(130,130,150,0.4)",width=1.5,shape="hv"),showlegend=False))
    for s,c in COLORS.items():
        mask=df["stage"]==s
        fig.add_trace(go.Scatter(x=df.loc[mask,"datetime"],y=df.loc[mask,"stage"],
            mode="markers",marker=dict(color=c,size=4,symbol="square"),name=STAGES[s]))
    fig.update_layout(
        xaxis_title="Uhrzeit",
        yaxis=dict(tickvals=[0,1,2,3],ticktext=["Tief","Leicht","REM","Wach"],
                   autorange="reversed",gridcolor="rgba(200,200,200,0.15)"),
        height=320, legend=dict(orientation="h",yanchor="bottom",y=1.02,x=0),
        margin=dict(l=10,r=10,t=10,b=10),
        plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(200,200,200,0.15)"))
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

all_paths   = tuple(str(p) for p in nights.values())
scaler, clf = get_model(all_paths)

df_raw  = load_features(str(selected_path))
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
    dist["Minuten"] = (dist["Epochen"]*EPOCH_SEC/60).round(1)
    fig_pie = px.pie(dist,values="Minuten",names="Phase",
        color="Phase",color_discrete_map={v:COLORS[k] for k,v in STAGES.items()},hole=0.4)
    fig_pie.update_layout(margin=dict(l=0,r=0,t=10,b=0),
        paper_bgcolor="rgba(0,0,0,0)",legend=dict(orientation="h"))
    st.plotly_chart(fig_pie, use_container_width=True)

with col_r:
    st.subheader("Feature-Wichtigkeit")
    imp = pd.DataFrame({"Feature":FEATURE_COLS,"Wichtigkeit":clf.feature_importances_})\
            .sort_values("Wichtigkeit",ascending=True)
    fig_bar = px.bar(imp,x="Wichtigkeit",y="Feature",orientation="h",
        color="Wichtigkeit",color_continuous_scale=["#74C0FC","#3B5BDB"])
    fig_bar.update_layout(margin=dict(l=0,r=0,t=10,b=0),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,xaxis=dict(gridcolor="rgba(200,200,200,0.15)"))
    st.plotly_chart(fig_bar, use_container_width=True)

if len(nights) > 1:
    st.divider()
    st.subheader("Alle Aufnahmen im Vergleich")
    rows = []
    for name, path in nights.items():
        df_n = predict_stages(load_features(str(path)), scaler, clf)
        rows.append({"Aufnahme": name, **compute_metrics(df_n)})
    st.dataframe(pd.DataFrame(rows).set_index("Aufnahme"), use_container_width=True)
