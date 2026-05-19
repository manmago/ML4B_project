from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SLEEPDATA_DIR = DATA_DIR / "sleepdata"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
PREPROCESSED_NIGHTS_DIR = PROCESSED_DIR / "nights"
FEATURE_CACHE_DIR = PROCESSED_DIR / "features"

DEFAULT_TIMEZONE = "Europe/Berlin"
DEFAULT_WINDOW_SECONDS = 120
DEFAULT_STEP_SECONDS = 60
DEFAULT_MERGE_TOLERANCE_MS = 20

SENSOR_FILE_MAP = {
    "Accelerometer.csv": "accelerometer",
    "Gyroscope.csv": "gyroscope",
    "TotalAcceleration.csv": "total_acceleration",
    "AccelerometerUncalibrated.csv": "accelerometer_uncalibrated",
    "GyroscopeUncalibrated.csv": "gyroscope_uncalibrated",
}
