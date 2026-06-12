# Raw data files

Contains the exported folders from each recorded sleep session with Sensor Logger.
Each folder is one night, named in the format `yyyy-mm-dd_hh-mm-ss`.

Each night folder contains these CSV files (as exported by Sensor Logger):

- `Accelerometer.csv` — calibrated acceleration
    - time, seconds_elapsed, z, y, x
- `Gyroscope.csv` — calibrated angular velocity
    - time, seconds_elapsed, z, y, x
- `Annotation.csv` — user-entered notes during the recording
    - time, seconds_elapsed, text, millisecond_press_duration
- `Metadata.csv` — recording metadata
    - version, device name, recording epoch time, recording time, recording timezone, platform, appVersion, device id, sensors, sampleRateMs, standardisation, platform version
- `AccelerometerUncalibrated.csv` — raw uncalibrated acceleration (not used by the pipeline)
- `GyroscopeUncalibrated.csv` — raw uncalibrated angular velocity (not used by the pipeline)
- `TotalAcceleration.csv` — computed total acceleration magnitude (not used by the pipeline)

The pipeline only reads `Accelerometer.csv` and `Gyroscope.csv` (plus `Annotation.csv`/`Metadata.csv`
where relevant); the uncalibrated and total-acceleration files are redundant and ignored.

Text input from `Annotation.csv` is manually transferred into `manual-labels.csv` (one row per night,
columns `night_id,bed_time,wake_time`), which provides fallback sleep/wake labels when smartwatch
tracking is unavailable.

Timestamps (sensor CSVs): Unix format (nanoseconds)
Timestamps (`manual-labels.csv`): `YYYY-MM-DD HH:MM:SS` in GMT+2 (ISO 8601)
Values: SI units (m/s² for acceleration, rad/s for angular velocity)

The raw night folders are large (~300MB each) and are ignored by git; only `manual-labels.csv` is tracked.
