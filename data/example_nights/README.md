# Example Nights

Put a small set of curated example nights here for the Streamlit demo.

The committed demo format is a compact `.joblib` bundle at the top level of this folder, for example `DEMO.joblib`.

Reference labels for the demo night live in `manual-labels.csv` at the top level of this folder. The app loads them the same way it loads `data/raw/manual-labels.csv`, so the demo night can show predicted-vs-real label comparisons.

Raw sensor folders such as `DEMO/` are useful when regenerating the bundle locally, but they should not be committed to GitHub.

Supported formats:

- A cached night bundle saved as `.joblib` for the demo app
- A raw night folder with the same structure as the data in `data/raw/` for local regeneration only

If you use a raw folder locally, it should contain files such as:

- `Accelerometer.csv`
- `Gyroscope.csv`
- `Metadata.csv`
- `Annotation.csv`

For the shared demo bundle, prefer the compact `.joblib` file so the app stays fast and the repository stays small.

## Current demo set

- **`DEMO.joblib`** — an accelerometer **+ gyroscope** Sensor Logger recording (2026-05-13). Used by the
  original binary V1 model (64 features) and the heuristic phase mode. Its reference bed/wake intervals come
  from `manual-labels.csv` (shared with `app-demo-night/`). This night was **not** used to train any model.
- **`2026-05-14_00-17-20.joblib`** — accelerometer **only**, built from `Accelerometer_Night_2.csv`, one of
  the 5 nights used to train the Samsung-labeled phase and binary models (full 100% Samsung stage coverage).
  Included so the "Sleep phases (trained)" and Samsung binary modes can be cross-verified against real
  training data with a true Samsung ground-truth overlay. **Caveat:** because it is a training night, its
  per-night predicted-vs-actual match is *in-sample* (optimistic) — the leave-one-night-out validation
  metrics shown in the app are the honest measure. The signal is downsampled to ~25 Hz so the bundle stays
  under GitHub's 100 MB limit. The original 64-feature binary model cannot run on this accel-only night
  (the app shows a guidance message; use the accelerometer-only Samsung model instead).