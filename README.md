# Sleep Classification and Hypnogram with Machine Learning

**Contributors:** [@Sieberuni](https://github.com/Sieberuni), [@manmago](https://github.com/manmago)

**Looking for the full scientific write-up** (motivation, methodology, results, limitations)? See [project.md](project.md).

---

This project is an app that turns **phone-sensor movement data into a picture of a night's
sleep**. Movement is recorded with a phone's accelerometer (and, in some recordings, a
gyroscope), chopped into short overlapping time windows, summarized into per-window
statistics, and fed to a **Random Forest** classifier. The predictions are drawn as a
hypnogram in an interactive **Streamlit** app. We started with binary sleep–wake
classification and later expanded the scope to explore the finer sleep phases (light, deep,
and REM).

The Streamlit demo reads compact example-night bundles from `data/example_nights/` so it can
load and display demo nights without shipping the large raw CSV recordings to GitHub.

## Project structure

```
ML4B_project/
├── app/
│   └── app.py              # The Streamlit app (all four modes live here)
├── src/
│   ├── main.py             # CLI entry point (python -m main <command>)
│   └── ml4b/               # The reusable pipeline package:
│       ├── io.py           #   load raw Sensor Logger CSVs / .joblib bundles, parse labels
│       ├── labels.py       #   resolve smartwatch + manual annotations into AWAKE/SLEEP
│       ├── preprocess.py   #   merge accelerometer + gyroscope streams, compute magnitudes
│       ├── features.py     #   slide windows, compute per-window statistics (64 features)
│       ├── model.py        #   binary Random Forest: train, evaluate, persist
│       ├── phases.py       #   4-class (Awake/Light/Deep/REM) phase model
│       ├── samsung.py      #   parse Samsung Health sleep-stage labels
│       ├── pipeline.py     #   orchestration (build dataset, train, predict a night)
│       └── config.py       #   all paths, window/step defaults
├── bin/                    # Unused files, kept for transparency
│                           #   (merged into the code above rather than used directly)
├── models/                 # Trained model bundles (.joblib)
├── data/
│   ├── raw/                # Raw per-night Sensor Logger recordings (gitignored)
│   ├── example_nights/     # Compact demo bundles used by the app (DEMO.joblib, …)
│   ├── processed/          # Cached intermediate night frames + feature tables
│   ├── samsung_sleep.csv   # Samsung Health sleep-stage export (ground-truth labels)
│   └── sleepdata/          # Huawei Health JSON exports (alternate label source)
├── notebooks/              # Exploration notebooks (data prep)
├── project.md              # Full scientific report
└── README.md               # You are here
```

## Using the app (online)

Open our [Streamlit app](https://ml4b-sleep-classification.streamlit.app/) and pick a view
with the **Model mode** selector in the sidebar. There are **four** modes.
We recommend starting with the binary mode, as it delivers the most reliable results:

- **Early version: Binary classification** — Classifies sleep vs. wake based on movement. 
  Original model that we started with to gain a better understanding. This model works 
  reasonably well on coarse in-bed labels and already provides practical value. 
  You can upload your own night here.
- **Sleep phases (trained)** — The **honest** 4-stage version, trained on **real Samsung
  Health sleep stages** (Awake / Light / Deep / REM). Its headline result: using phone motion 
  alone, 4-stage classification scores about **0.24 balanced accuracy** (random guessing ≈ 0.25) 
  — confirming that fine sleep stages cannot be reliably distinguished from movement data 
  alone without additional signals like heart rate.Shows the model's validation metrics 
  and confusion matrix, and lets you **upload your own night**.
- **Sleep phases** — A polished demo dashboard for 4-stage visualization. This mode uses 
  heuristic pseudo-labels generated at runtime and is intended for illustration only, 
  not as validated predictions.
- **ℹ️ About** — A plain-language page explaining the views and how to read the charts.

Note: The online version runs on compressed demo data, which slightly reduces feature quality. 
For the best results with your own recordings, run the app locally.

A couple of caveats for the online version: both analysis modes run on a GitHub-friendly,
compressed `.joblib` demo night (used only for validation, never for training), which
slightly reduces the quality of some features.

If you want to validate against **your own** sleep data and get the best results, run the
app **locally** (see below) — local runs are faster and not subject to the cloud's upload
size limits.

## Running locally

### Prerequisites

- Python 3.11+
- The [uv](https://docs.astral.sh/uv/) package manager
- Git

### 1. Clone the repository

```bash
git clone https://github.com/manmago/ML4B_project.git
cd ML4B_project
```

### 2. Set up the virtual environment

**Windows (PowerShell — the VS Code terminal default):**

```powershell
# One-time only, if scripts are blocked:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

uv sync
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**

```cmd
uv sync
.venv\Scripts\activate.bat
```

**Mac/Linux:**

```bash
uv sync
source .venv/bin/activate
```

### 3. Run the app

```bash
streamlit run app/app.py
```

The app works out of the box using the demo bundles in `data/example_nights/`. Locally you
can also upload your own night (a `.zip` containing `Accelerometer.csv` and `Gyroscope.csv`,
as exported by the [Sensor Logger](https://www.tszheichoi.com/sensorlogger) app) for the
binary mode, or an `Accelerometer.csv`/`.zip` for the trained phase mode.

### Optional: the CLI

The pipeline can also be driven from the command line (run from `src/`):

```bash
python -m main train               # train the binary AWAKE/SLEEP model
python -m main train-phases        # train the 4-class Samsung-labelled phase model
python -m main train-binary-samsung  # train a Samsung-labelled wake/sleep model
python -m main build               # build the labeled feature dataset to CSV
python -m main preprocess          # build/cache labeled night frames + features
python -m main predict             # print a hypnogram preview for one night
```

Most commands accept `--raw-dir`, `--sleepdata-dir`, `--window-seconds`, and `--step-seconds`
(defaults: 120 s windows / 60 s steps). Training the phase models requires the raw
accelerometer nights to be present locally.
