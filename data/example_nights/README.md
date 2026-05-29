# Example Nights

Put a small set of curated example nights here for the Streamlit demo.

The committed demo format is a compact `.joblib` bundle at the top level of this folder, for example `DEMO.joblib`.

Raw sensor folders such as `DEMO/` are useful when regenerating the bundle locally, but they should not be committed to GitHub.

Supported formats:

- A cached night bundle saved as `.joblib` for the demo app
- A raw night folder with the same structure as the data in `data/raw/` for local regeneration only

If you use a raw folder locally, it should contain files such as:

- `Accelerometer.csv`
- `Gyroscope.csv`
- `Metadata.csv`
- `Annotation.csv`

Keep these example nights separate from training and validation nights so they do not leak into evaluation. For the shared demo bundle, prefer the compact `.joblib` file so the app stays fast and the repository stays small.