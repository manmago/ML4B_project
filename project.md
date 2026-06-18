# 1 Introduction

## Motivation

In Germany, roughly 20% of the population posess a smartwatch or similar device of which majority use it to track health data (Manz et al., 2025).

Sleep tracking contributes to overall sleep quality, health, and efficiency and increases the well-being of individuals (Feng et al., 2026).

However, certain demographic groups, such as older people, people with lower income and lower physical activity levels are less likely to own a wearable device and use it for health tracking reasons (Manz et al., 2025).

According to the Federal Statistical Office, roughly 98% of households own a smartphone (Statistisches Bundesamt, 2022), which presents an opportunity to introduce a bigger part of the population to accessible sleep tracking. Despite this, mobile apps made for this purpose lack empirical evidence (Amanth, 2021) and from personal experience, reliable results. 

We aim to address this gap by training a ML model to classify sleep states. 

## Research Questions

1. Can smartphone accelerometer and gyroscope data, collected non-intrusively during nighttime, reliably classify binary sleep states (asleep vs. awake) with at least 75% accuracy?

2. Can we estimate subjective sleep quality and total sleep duration from smartphone sensor patterns, validated against smartwatch sleep tracking data?

3. Can smartphone sensor data, in combination with derived sleep quality metrics, classify sleep phases (light sleep, deep sleep, potentially REM) with meaningful accuracy, and what are the physiological limitations of such classification without direct biometric signals?

This document follows the standard CRISP-DM-style structure of an ML project report: motivation and research questions, related work, methodology (data understanding, preparation, modeling, and evaluation), results, discussion (including limitations and ethics), conclusion, and sources.

# 2 Related Work

*Golden standard* of sleep tracking: **Polysomnography (PSG)**. Monitors brain activity, eye and muscle movements, heart rate, oxygen saturation, airflow, and respiratory effort (Rundo & Downey, 2019).  
Researchers trained a deep neural network for sleep classification using cardiorespiratory and body movement signals (Morokuma et al., 2023), and a related study used multi-modality signals for sleep staging (Satapathy et al., 2024).

*Wearables* rely on a combination of body movement, electrocardiogram data, blood volume changes, oxygen saturation, and microphone data to predict sleep and sleep phases (Manz et al., 2025).

*Mobile* applications are more limited in clinical sensor data, therefore lean on audio data or movement measured with accelerometer and gyroscope. Additionally, almost all apps on the market lack empirical evidence, for instance validation against the golden standard, PSG. Apps with such validation studies present weak correlation. One example is the Sleep Cycle application, which claims to use AI-powered analysis technologies (Amanth, 2021; Sleep Cycle, n.d.).

A well-documented *GitHub project* is the [sleep-tracker](https://github.com/josephbima/sleep-tracker), using just the accelerometer and a Random Forest.   
[The most recent project](https://github.com/mkucukos/sleep-awake-detection) from 2025 utilizes wearables and applies Logistic Regression, Random Forest,and XGBoost on accelerometer and gyroscope data, as well as body temperature, and heart rate.

There appears to be a gap in reliable and accessible mobile sensor-based sleep tracking. Despite not being able to validate our work against PSG standards, the challenge is to come close to the reliability of wearable devices.

# 3 Methodology

## 3.1 General Methodology

The project did not undergo a linear process, there was a lot of jumping back and forth between different phases. Code was continuously changed. The best documentation of the process would be the activity tab.

**Recording**  
We used the Sensor Logger mobile application to record sensor data. Since sleep phases correlate with movement, we decided to use the accelerometer and gyroscope sensors to record our movement during sleep.   
The recordings were started 30mins before deciding to put away the phone and go to sleep and lasted the entire night until we would wake up and turn off the recording approx. 10mins later. This was due to our first phase, the binary sleep-wake detection, so that the first models had enough sensor data of wake stages.  
The recordings were contributed by two team members: [@manmago](https://github.com/manmago) recorded the 5 Huawei-labeled accelerometer + gyroscope nights used for the binary model, and [@Sieberuni](https://github.com/Sieberuni) recorded the 5 Samsung-labeled accelerometer-only nights used for the phase and Samsung-binary models. 

**Exploration**
Due to time pressure, exploration in notebooks/* is limited to mostly data preparation.

**Labeling**  
Labeling for the binary model went different than planned. Contributor [@manmago](https://github.com/manmago) used a Huawei SmartWatch and the associated Health App to track sleeping stages to use for labeling. Unfortunately, when requesting the data, only a small selection of the nights needed were provided. The rest of the recorded nights with sleep classification from the app were never delivered, even after multiple requests.  
Therefore, a fallback mechanism was used and merged with the labels from the health app provider. It's less rigorous, however the times of "going to sleep" and "waking up" were manually added in the Annotations.csv and later transferred to a uniform format. This will further be discussed in the limitations chapter.  
For the later sleep-phase track, contributor [@Sieberuni](https://github.com/Sieberuni) provided a separate set of nights labeled with a **Samsung Health `sleep_stage` export**. This gives genuine per-stage ground truth (Awake/Light/Deep/REM) rather than just bed/wake intervals: the `ml4b.samsung` module parses the export into stage intervals and assigns each feature window the stage it overlaps most in time (maximum-overlap labeling). These real per-window labels are what made the *honest* phase and wake/sleep evaluation in the Results possible. 

**Modeling**
For modeling, the best fit based on related work seemed to be a Random Forest. We first trained the binary model with the Huawei-labeled nights recorded by [@manmago](https://github.com/manmago) and the corresponding labeling. The resulting `models/sleep_model_w120_s60.joblib` was shown, evaluated and tweaked with in the streamlit app which is the binary mode you can see in the current app.  
Once @Sieberuni's Samsung-labeled nights became available, two further Random Forests were trained on them: `models/sleep_phase_model_w120_s60.joblib`, a 4-class (Awake/Light/Deep/REM) phase model, and `models/sleep_model_samsung_w120_s60.joblib`, a binary wake/sleep model on the same nights with the stages collapsed to AWAKE vs. SLEEP. Both use the accelerometer-only feature set (32 features, no gyroscope). Separately, the app's exploratory "Sleep phases" mode trains a fourth Random Forest at runtime on **heuristic** activity-percentile pseudo-labels (not ground truth) purely to illustrate a 4-stage hypnogram. In total: three trained, saved models plus one runtime heuristic model.

## 3.2 Data Understanding and Preparation

### Dataset and structure

Primarily smartphone sensor data from gyroscope and accelerometer. For labeling, a hybrid approach with smartwatch-based sleeptracking data and manual annotations was used. Sampling rate set to 100Hz.

The project uses two separate datasets, one per track:

- **Binary track (Huawei, accelerometer + gyroscope):** 5 nights (`n_groups=5` in the cached feature dataset). After windowing (120s windows, 60s steps), this yields 2245 labeled feature windows, each described by 64 features. At a 60s step size, 2245 windows correspond to roughly 37 hours of combined recording, i.e. an average of about 7.5 hours per night - in line with the recording protocol described above (full night plus ~40 minutes of buffer at each end).
- **Phase track (Samsung, accelerometer only):** 5 further nights labeled with Samsung Health stages, yielding 1747 labeled windows. These nights were recorded without the gyroscope, so each window has 32 accelerometer-only features instead of 64. This dataset feeds the 4-class phase model and the Samsung wake/sleep model.

Each night equals a .csv file including files: 
- Accelerometer.csv - Calibrated acceleration (x, y, z axes)
- AccelerometerUncalibrated.csv - Raw uncalibrated acceleration
- Gyroscope.csv - Calibrated angular velocity (x, y, z axes)
- GyroscopeUncalibrated.csv - Raw uncalibrated angular velocity
- TotalAcceleration.csv - Computed total acceleration magnitude
- Metadata.csv - Recording metadata (device info, sampling rate, timezone)
- Annotation.csv - User-provided annotations (manual sleep-wake notes)

Columns: time, seconds_elapsed, x, y, z  
Timestamps: Unix format (nanoseconds)  
Values: SI units (m/s² for acceleration, rad/s for angular velocity)

**Specialities**  
Two labeling strategies are used across the two tracks. For the binary track, hybrid labeling with Huawei smartwatch sleep intervals as the primary source, and manual annotations as a fallback when smartwatch tracking is unavailable. For the phase track, real per-window stage labels from a Samsung Health `sleep_stage` export (Awake/Light/Deep/REM), assigned to windows by maximum time overlap.

### Dataset preparation

1. Data cleaning
- Ignore or remove redundant files: TotalAcceleration.csv, AccelerometerUncalibrated.csv, GyroscopeUncalibrated.csv

Exploration notebook findings showed mostly synchronized and stable sensor data. To avoid over-cleaning:
- Invalid values are coerced to missing with `pd.to_numeric(..., errors="coerce")` and only dropped when they cannot be aligned or featurized.
- Outliers are not aggressively filtered because the window-based features are already fairly robust to short spikes, and real sleep movement can look unusual.
- Duplicate-specific cleaning was not needed for the inspected nights.
- Sensor errors are handled by dropping only samples or windows that remain unusable after alignment and feature extraction.
- Smartwatch wake/sleep failure is corrected by manual annotation.
2. Preprocessing
- Sensors merged per night with nearest timestamp sensitive method due to few mismatches.
3. Feature selection
- Each 120s window with a 60s step is described by 8 summary statistics (mean, std, min, max, median, IQR, energy, range) computed over the available signal channels. For the binary (Huawei) track these are 8 channels (accelerometer x/y/z/magnitude and gyroscope x/y/z/magnitude), giving 64 numeric features per window. The Samsung phase track is accelerometer-only (no gyroscope was logged), so it has 4 channels and 32 features per window.
- The training pipeline (`ml4b.model.create_model_pipeline`) adds a `SelectFromModel` step: a separate `RandomForestClassifier` ranks all features by importance, and only those at or above the median importance are kept (around half) before they reach the final classifier. This reduces redundancy between highly-correlated statistics (e.g. energy vs. std of the same channel) and keeps the model focused on the most informative signals.
- The selected feature names and their count are recorded in `model_bundle.metadata["selected_feature_names"]` / `metadata["n_selected_features"]` for inspection after training.
4. Data splitting
- Models are evaluated with `GroupKFold` cross-validation grouped by `night_id`, using up to 5 folds (one per recorded night). This is effectively a leave-one-night-out evaluation: in each fold, the model is trained on 4 nights and validated on the remaining, unseen night, which gives a realistic estimate of how the model generalizes to a new night/person.
- After cross-validation, a final model is fit on all available windows from all 5 nights for use in the app.
5. Potential bias discussion
- Time of the day: The training inputs are motion sensors only and window_start, window_end, label, sleep_fraction, and sample_count are explicitly excluded from the features. However, due to a mostly regular sleep schedule, a slight bias in sleep recognition is possible.  
 

## 3.3 Modeling and Evaluation

**Selected model architecture:**
- Random Forest throughout, in four concrete forms:
  1. Binary AWAKE/SLEEP on Huawei accel+gyro nights (`sleep_model_w120_s60`) — the validated centerpiece.
  2. 4-class Awake/Light/Deep/REM on Samsung accel-only nights (`sleep_phase_model_w120_s60`).
  3. Binary AWAKE/SLEEP on the same Samsung accel-only nights (`sleep_model_samsung_w120_s60`).
  4. An exploratory 4-class model trained at runtime in the app on heuristic activity-percentile pseudo-labels (not validated, illustrative only).

**Training approach:**
- Train the supervised models (1-3 above) on window-based features derived from the merged night sensor data
- Use the same preprocessing and feature extraction pipeline for training and prediction
- The heuristic mode (4) reuses the same feature pipeline but generates its own labels at runtime, so it has no ground-truth evaluation by design

**Evaluation metrics:**
- Accuracy: simple overall baseline
- Balanced accuracy: useful when sleep states are uneven
- F1: useful for binary classification because it balances precision and recall
- Precision: reliability of predicted sleep
- Recall: how much true sleep is recovered
- ROC-AUC: quality across thresholds

# 4 Results

**Artifacts**

- `src/ml4b`: a reusable Python package implementing the full pipeline - `io` (loading raw Sensor Logger CSVs / `.joblib` night bundles), `preprocess` (merging accelerometer and gyroscope streams, computing magnitudes), `labels` (resolving smartwatch and manual annotation labels into AWAKE/SLEEP intervals), `features` (window-based feature extraction, 64 features per window), `model` (training, feature selection, evaluation, persistence), and `pipeline` (running the full pipeline on a new night to produce a per-window hypnogram).
- `models/sleep_model_w120_s60.joblib`: the original binary AWAKE/SLEEP Random Forest model bundle (accelerometer + gyroscope, 64 features), including its feature columns, cross-validated metrics, and metadata.
- `models/sleep_phase_model_w120_s60.joblib`: a 4-class (Awake/Light/Deep/REM) Random Forest trained on **real Samsung Health sleep-stage labels** over five accelerometer-only nights. Built by the `ml4b.samsung` (label parsing) and `ml4b.phases` (multiclass model) modules and the `train-phases` CLI command.
- `models/sleep_model_samsung_w120_s60.joblib`: a binary AWAKE/SLEEP Random Forest trained on the same five accelerometer-only nights, with the Samsung stages collapsed to wake vs. sleep (32 accelerometer features, no gyroscope). Built via the `train-binary-samsung` CLI command and selectable in the app's binary mode.
- `data/example_nights/DEMO.joblib`: a compressed example night (full 100Hz accelerometer + gyroscope data plus labels) bundled with the repository so the app works out of the box without any local raw data.
- `data/samsung_sleep.csv`: a Samsung Health `sleep_stage` export providing ground-truth 4-stage labels (Awake/Light/Deep/REM) for the recorded nights, used to train and evaluate the phase model.
- `app/app.py`: a Streamlit app with four modes (described below).
- A `main.py` CLI entry point for training the models (`uv run python -m main train ...` for the binary model, `train-phases` for the 4-class phase model, `train-binary-samsung` for a Samsung-labelled wake/sleep model).

**Libraries and tools**

- `pandas` / `numpy` for data loading, merging, and feature computation.
- `scikit-learn` for the modeling pipeline (`Pipeline`, `SimpleImputer`, `SelectFromModel`, `RandomForestClassifier`, `GroupKFold`, `cross_validate`, evaluation metrics).
- `joblib` for serializing the trained model bundle and the compressed example night.
- `streamlit` for the interactive web app, with `plotly` for the magnitude and hypnogram charts.
- `uv` for dependency and environment management.

**App concept**

The Streamlit app has four modes, selectable from the sidebar (three analysis modes plus an information page):

- *"Early version: Binary classification"* (the centerpiece of the project): loads the trained `sleep_model_w120_s60.joblib` model and runs it on the bundled `DEMO` example night, showing sensor magnitude plots, a 10-minute hypnogram of predicted sleep probability vs. the reference label (where available), and a table of the raw per-window predictions. It also lets users **upload their own night** as a `.zip` containing `Accelerometer.csv` and `Gyroscope.csv` (the same format produced by the Sensor Logger app); the upload is unpacked into a temporary directory, run through the same merging/feature-extraction/prediction pipeline, and rendered with the same charts - all in-memory, for inference only, with no retraining and no persistence.
- *"Sleep phases"*: an exploratory mode that reuses the same window pipeline, but trains a 4-stage (deep/light/REM/awake) Random Forest at runtime on heuristic, activity-percentile-based pseudo-labels. It shows a hypnogram, sleep-phase distribution, and feature-importance chart, with a clear disclosure that its labels are heuristic rather than ground-truth.
- *"Sleep phases (trained)"*: loads the saved `sleep_phase_model_w120_s60.joblib` model, which **was** trained on real Samsung Health stage labels, and predicts stages for the selected night, overlaying the prediction against the Samsung ground truth where available. Crucially, this mode reports the model's honest leave-one-night-out validation metrics and confusion matrix - which show that accelerometer-only data is not sufficient for reliable 4-class staging (see below). It also supports uploading your own accelerometer recording.
- *"ℹ️ About"*: a plain-language information page that explains the four views, how to read the hypnogram/distribution/confusion-matrix charts, and the example nights - so a first-time visitor can orient themselves without reading this report.

**Results on unseen data**

*Original binary model (accelerometer + gyroscope, coarse in-bed labels).* The first binary AWAKE/SLEEP model was evaluated with leave-one-night-out (`GroupKFold`, 5 folds) cross-validation across the 5 recorded nights (2245 windows total, 64 engineered features). The averaged out-of-fold metrics were:

| Metric | Value |
| --- | --- |
| Accuracy | 0.985 |
| Balanced accuracy | 0.953 |
| F1 | 0.992 |
| Precision | 0.992 |
| Recall | 0.991 |
| ROC-AUC | 0.995 |

These numbers look excellent, but they must be read with care: the labels for this model came from manual/smartwatch **in-bed intervals**, i.e. the whole bed-to-wake span is stamped SLEEP and everything outside it AWAKE. That is essentially an *in-bed vs. out-of-bed* task, which accelerometer data separates easily (the user is clearly moving around before and after), so the high scores overstate true sleep-staging ability. The harder, more honest evaluation below uses per-window Samsung Health labels that mark the brief awakenings scattered *throughout* the night.

*4-class phase model (accelerometer only, real Samsung Health stage labels).* Using the five new accelerometer nights labelled with genuine Samsung Health hypnogram stages (1747 labelled windows: 1206 Light, 213 REM, 180 Deep, 148 Awake; 32 accelerometer features, no gyroscope), a 4-class Random Forest was trained and evaluated with the same leave-one-night-out protocol. The result is a clear **negative finding**:

| Metric | Value |
| --- | --- |
| Balanced accuracy | 0.24 (4-class chance ≈ 0.25) |
| Macro F1 | 0.21 |
| Light recall | 0.95 |
| Deep / REM / Awake recall | ≈ 0 |

The model collapses to predicting the majority "Light" class and cannot recover Deep, REM, or Awake on held-out nights. This is not a software bug (labels and features were verified) but a **physiological limitation**: phone-accelerometer motion alone does not carry enough signal to separate sleep stages that differ mainly in brain/cardiac activity rather than gross movement. This directly answers research question 3 in the negative.

*Binary wake/sleep on the same real labels (accelerometer only).* Collapsing the Samsung stages to AWAKE (Awake) vs. SLEEP (Light/Deep/REM) and retraining the binary model (1747 windows, 1599 SLEEP / 148 AWAKE) gives:

| Metric | Value |
| --- | --- |
| Accuracy | 0.93 |
| Balanced accuracy | 0.60 |
| F1 (SLEEP) | 0.96 |
| Precision | 0.93 |
| Recall | 1.00 |
| ROC-AUC | 0.57 |

Here the headline accuracy (0.93) is again an imbalance artifact - 91.5% of windows are SLEEP, and the model achieves it by predicting SLEEP almost unconditionally (recall ≈ 1.0). The honest discriminative metrics, balanced accuracy 0.60 and ROC-AUC 0.57, are only marginally above chance: detecting the short wake epochs *inside* a night from phone motion is hard, because lying awake but still looks like sleep to an accelerometer. This is a much more realistic picture of the task than the 0.985 from the in-bed-labelled model, and reframes that earlier result accordingly.

# 5 Discussion

The project produced a working artifact with a clear, honest result: binary asleep-vs-awake classification from phone motion works well on coarse in-bed labels, while full 4-class sleep-stage classification from the same signal hits a physiological ceiling and performs near chance. The sections below discuss what this means, where the artifact is limited, and the ethical and societal considerations around it.

**Limitations**
High initial expectations met real-world data and process constraints partway through the project, which shaped what the final artifact could achieve. The main limitations, several of which were addressed over the course of the project, are:
1. **Grounded labeling data.** Limited access to ground-truth sleep labels was the project's biggest early constraint. It is now substantially addressed: a Samsung Health `sleep_stage` export (`data/samsung_sleep.csv`) provides real per-window 4-stage ground truth for five accelerometer nights, which let us train and *honestly evaluate* both a 4-class phase model and a wake/sleep model (see Results). The grounded labels did not make the models accurate - instead they revealed a hard ceiling (item 5) - but evaluation is no longer a blind spot.
2. **Recording size.** Raw nights are ~300MB each, so sample and demo sizes are modest. The raw demo night had to be converted into a compressed `DEMO.joblib` bundle because the uncompressed version was too large for GitHub; the 3x-compressed version keeps it below the hard 100MB limit. Git LFS was tried as an alternative but caused complications with the Streamlit app.
3. **Uploading your own night.** The Streamlit app now supports analyzing a recording directly: the "Early version: Binary classification" mode includes an "Analyze your own night" upload feature accepting a `.zip` with `Accelerometer.csv` and `Gyroscope.csv` (the trained phase mode accepts accelerometer-only uploads). Streamlit Community Cloud caps uploads at 200MB, so very long nights (~300MB) may still need to be analyzed locally.
4. **Dataset breadth.** The dataset remains modest but has grown. Five additional nights with matching Samsung Health stage labels are now available for training/evaluation (kept local, ~280MB each, gitignored). They are accelerometer-only - the phone was not logging the gyroscope - so the phase model uses 32 features instead of 64 and cannot reuse the gyroscope-dependent binary model.
5. **Physiological ceiling of accelerometer-only staging (key finding).** With real Samsung labels, the 4-class model performs at roughly chance (balanced accuracy ≈ 0.24) and the wake/sleep model only marginally above it (balanced accuracy ≈ 0.60, ROC-AUC ≈ 0.57). Phone motion alone cannot reliably distinguish sleep stages or catch brief in-night awakenings without complementary biometric signals (heart rate, HRV, respiration). This is the central honest result of the project and bounds what any similar accelerometer-only app can achieve.
6. **Bed and sleeper variation.** The sensor data was recorded on two beds/mattresses with no other person present, so accuracy on different mattresses or with a bed partner is untested and results could vary considerably under other circumstances.

**Ethics, effects on society and environment**
At best, the app could provide beneficial health data for better living and lower the barrier to sleep tracking for people who do not own a smartwatch. The app now supports uploading a recorded night (a `.zip` with `Accelerometer.csv` and `Gyroscope.csv`) for analysis. This data is processed entirely in-memory for the duration of the session: it is extracted to a temporary directory, run through the prediction pipeline, and then discarded. Nothing is persisted to disk beyond the temporary extraction, sent to any external service, or used to retrain the model. No accounts, identifiers, or location data are collected, so the privacy risk of using the app is low - though users should still be mindful that motion sensor data, while not biometric in the medical sense, can in principle reveal information about a person's daily routine if it were ever stored or shared.

**Danger**
Discrimination could happen in people who are limited in movement or have a lower body weight/mattress hardness ratio. They could experience worse results, since the training data only covers two beds/mattresses and no participants with movement impairments. Generalization to wheelchair users, people sharing a bed with a partner or pet, or very different mattress types is untested and likely worse.

**Transparency**
The app explicitly distinguishes between its three analysis modes: the binary model (with reported cross-validation metrics), the exploratory "Sleep phases" mode (clearly labeled as heuristic, non-validated pseudo-labels), and the "Sleep phases (trained)" mode, which shows a real-label model alongside its honest, weak validation metrics and confusion matrix rather than only a polished hypnogram. A dedicated "About" page summarizes all of this in plain language. We deliberately surface the negative result instead of hiding it behind a plausible-looking chart. Users are told what data is required, how it is processed, and that uploaded data is not stored.

For broader context on the societal effects of automated decision-making and health-tracking technology, see the AlgorithmWatch "Automating Society" reports (https://algorithmwatch.org/en/) and the publications of the AI Now Institute (https://ainowinstitute.org/).

**Further Research**
- Frequency-domain features (e.g. FFT-based spectral power per band) in addition to the current time-domain statistics, which could better separate sleep phases that differ mainly in movement frequency rather than amplitude.
- Systematic hyperparameter tuning (e.g. grid/random search over `n_estimators`, `max_depth`, `min_samples_leaf`, and the `SelectFromModel` threshold) instead of the currently fixed Random Forest configuration.
- Fusing the accelerometer with biometric signals (heart rate, HRV, respiration, SpO2) - the Samsung-labelled experiments show this is the missing ingredient: motion alone hit a hard ceiling for both 4-class staging and in-night wake detection, so richer phase classification almost certainly requires sensors beyond the phone.
- Re-recording nights with the gyroscope enabled (the new labelled nights are accelerometer-only), so the full 64-feature pipeline and the existing binary model can be applied to ground-truth-labelled data.
- Collecting a larger, multi-participant dataset across different mattresses, bed-sharing situations, and movement profiles to assess and improve generalization, and to make the bias and fairness discussion above empirically testable.

# 6 Conclusion

Based on movement during sleep, the app can estimate sleep and wake stages with a validated, cross-validated Random Forest model, and can also explore the different sleep phases (light, deep and REM) using the same engineered features with heuristic labels. The binary mode is the validated centerpiece, with reported metrics and an upload feature for analyzing your own night, while the multi-phase mode is an exploratory, clearly-labeled preview of what richer phase classification could look like once real per-phase labels become available.
As a first end-to-end attempt at building an ML-based app, the project delivered a clear and honest outcome rather than an inflated one: binary sleep-wake detection works well, while 4-class staging from phone motion alone runs into a physiological ceiling. Along the way we learned how to build a Streamlit app, how to record and label sensor data, and what it takes to train and *honestly evaluate* a model - including the value of reporting a negative result transparently.
The resulting models and app are intentionally scoped as a first version and demonstration rather than a finished product, and they provide a useful baseline and inspiration for similar projects to build on. 

# 7 Sources

Ananth S. (2021). Sleep apps: current limitations and challenges. Sleep science (Sao Paulo, Brazil), 1
(1), 83–86. https://doi.org/10.5935/1984-0063.20200036

Feng, S., Mäntymäki, M., & Pappas, I. O. (2026). Sleep tracking: An integrative review, conceptual framework and future research agendas. Behaviour & Information Technology, 0(0), 1–31. https://doi.org/10.1080/0144929X.2026.2621789


Manz, K., Krug, S., Kühnelt, C., Lemcke, J., Öztürk, I., & Loss, J. (2025). Consumer Wearable Usage to Collect Health Data Among Adults Living in Germany: Nationwide Observational Survey Study. JMIR mHealth and uHealth, 13, e59199. https://doi.org/10.2196/59199

Morokuma, S., Hayashi, T., Kanegae, M., Mizukami, Y., Asano, S., Kimura, I., Tateizumi, Y., Ueno, H.,Ikeda, S., & Niizeki, K. (2023). Deep learning-based sleep stage classification with cardiorespiratory and body movement activities in individuals with suspected sleep disorders. Scientific reports, 13(1), 17730. https://doi.org/10.1038/s41598-023-45020-7 https://pmc.ncbi.nlm.nih.gov/articles/PMC10584883/ 

Rundo, J. V., & Downey, R., 3rd (2019). Polysomnography. Handbook of clinical neurology, 160, 381–392 https://doi.org/10.1016/B978-0-444-64032-1.00025-4

Satapathy, S.K., Brahma, B., Panda, B. et al. Machine learning-empowered sleep staging classification using multi-modality signals. BMC Med Inform Decis Mak 24, 119 (2024). https://doi.org/10.1186/s12911-024-02522-2 https://link.springer.com/article/10.1186/s12911-024-02522-2

Sleep Cycle. (n.d.). The Sleep Cycle Advantage. https://sleepcycle.com/partnerships/value-of-sleep-cycle. Accessed 28.04.2026.

Statistisches Bundesamt (2022). Daten aus den Laufenden Wirtschaftsrechnungen (LWR) zur Ausstattung privater Haushalte mit Informationstechnik. https://www.destatis.de/DE/Themen/Gesellschaft-Umwelt/Einkommen-Konsum-Lebensbedingungen/Ausstattung-Gebrauchsgueter/Tabellen/a-infotechnik-d-lwr.html