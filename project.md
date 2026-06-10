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

- How is this document structured

# 2 Related Work

- What have others done in your area of work/ to answer similar questions?

*Golden standard* of sleep tracking: **Polysomnography (PSG)**. Monitors brain activity, eye and muscle movements, heart rate, oxygen saturation, airflow, and respiratory effort (Rundo & Downey, 2019).  
Researchers trained a deep neural network for sleep classification (Morokuma/Sata)

*Wearables* rely on a combination of body movement, electrocardiogram data, blood volume changes, oxygen saturation, microphone data to predict sleep, and sleep phases (Source tbd).

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
In total, contributor manmago recorded 5-10 nights. 

**Exploration**
Due to time pressure, exploration in notebooks/* is limited to mostly data preparation.

**Labeling**  
Labeling for the binary model went different than planned. Contributor manmago used a Huawei SmartWatch and the associated Health App to track sleeping stages to use for labeling. Unfortunately, when requesting the data, only a small selection of the nights needed were provided. The rest of the recorded nights with sleep classification from the app were never delivered, even after multiple requests.  
Therefore, a fallback mechanism was used and merged with the labels from the health app provider. It's less rigorous, however the times of "going to sleep" and "waking up" were manually added in the Annotations.csv and later transferred to a uniform format. This will further be discussed in the limitations chapter. 

**Modeling**
For modeling, the best fit based on related work seemed to be a Random Forest. We trained one model with the nights recorded from manmago and the corresponding labeling. The resulting model/sleep_model_w120_s60.joblib was shown, evaluated and tweaked with in the streamlit app which is the binary mode you can see in the current app.

## 3.2 Data Understanding and Preparation

### Dataset and structure

Primarily smartphone sensor data from gyroscope and accelerometer. For labeling, a hybrid approach with smartwatch-based sleeptracking data and manual annotations was used. Sampling rate set to 100Hz.

Total recorded: x nights, y duration

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
Hybrid labeling with smartwatch sleep intervals as the primary source. Manual annotations are used as a fallback when smartwatch tracking is unavailable.

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
4. Data splitting
5. Potential bias discussion
- Time of the day: The training inputs are motion sensors only and window_start, window_end, label, sleep_fraction, and sample_count are explicitly excluded from the features. However, due to a mostly regular sleep schedule, a slight bias in sleep recognition is possible.  
 

## 3.3 Modeling and Evaluation

**Selected model architecture:**
- Random Forest as the main supervised baseline

**Training approach:**
- Train the supervised models on window-based features derived from the merged night sensor data
- Use the same preprocessing and feature extraction pipeline for training and prediction

**Evaluation metrics:**
- Accuracy: simple overall baseline
- Balanced accuracy: useful when sleep states are uneven
- F1: useful for binary classification because it balances precision and recall
- Precision: reliability of predicted sleep
- Recall: how much true sleep is recovered
- ROC-AUC: quality across thresholds

# 4 Results

- Describe what artifacts you have build

- Describe the libraries and tools you use

- Describe the concept of your app

- Describe the results you achieve by applying your trained models on unseen data

- Descriptive Language (no judgement, no discussion in this section -> just show what you built)

# 5 Discussion

- Now its time to discuss your results/ artifacts/ app 

**Limitations**
High expectations in the beginning and issues in the middle of the project permanently changed the efficiency of the artifact.  
1. (Partially) missing "grounded" labeling data: This is the biggest limitation. Even the binary model is impaired due to this. Evaluation of the models' performance is challenging. 
2. Raw nights are ~300MB's big: Sample size and demo size is small. The raw demo night added had to be converted into a compressed `DEMO.joblib` bundle, because the uncompressed version was too large for GitHub. We ended up uploading the 3x compressed version, which keeps it below the hard 100MB limit. Git LFS as an alternative did not work, there were complications with the Streamlit app.
3. There is also no option to upload a night recorded to the Streamlit app: You would have to do this locally instead. 
4. Combined, only one night to demonstrate the app and models is available. This night also does not have health app sleep labels, which makes the project hard to evaluate. 
5. There seems to be a bug regarding the amount of time slept and in the different sleep phases. This is likely to the 3x compression of the uploaded DEMO.joblib file, whereas the original folder with .csv recordings show more realistic time. Therefore, the live/cloud Streamlit App performs worse than the local one.
5. Inaccuracy on different mattresses or with partners: The sensor data was recorded on 2 beds and mattresses with no other person present. Results could vary a lot given other circumstances. 

**Ethics, effects on society and environment**
There are no known concerns regarding ethics. At best, the app could provide beneficial health data for better living. Since there is no option to upload sleep data, privacy risk is non-existent. 
TODO

**Danger**
Discrimination could happen in people who are limited in movement or have a lower body weight/mattress hardness ratio. They could experience worse results.  

- Transparency 
- Possible sources https://algorithmwatch.org/en/ Have a look at the "Automating Society Report"; https://ainowinstitute.org/ Have a look at this website and their publications

- Further Research: What could be next steps for other researchers (specific research questions)

# 6 Conclusion

Based on movement during sleep, the app can estimate your sleep and wake stages. It can also predict the different sleep phases light, deep and REM. The binary mode shows an early experimental approach, while the multi-phase mode depicts a user-friendly UI and better  
Facing hardships, this project was an experimental first try at creating an ML-based app. The results did not meet the expectations, yet many things about project and processes were learned. We learned how to build a simple app in Streamlit and what it takes to train a model. 
The resulting models and app are limited in their intention and should not be perceived as final or functional, yet they serve their purpose as a first version or a demonstration of what a similar artifact should provide. 
The artifact we built can be used as a baseline or inspiration for similar projects. 

# 7 Sources

Ananth S. (2021). Sleep apps: current limitations and challenges. Sleep science (Sao Paulo, Brazil), 1
(1), 83–86. https://doi.org/10.5935/1984-0063.20200036

Feng, S., Mäntymäki, M., & Pappas, I. O. (2026). Sleep tracking: An integrative review, conceptual framework and future research agendas. Behaviour & Information Technology, 0(0), 1–31. https://doi.org/10.10800144929X.2026.2621789


Manz, K., Krug, S., Kühnelt, C., Lemcke, J., Öztürk, I., & Loss, J. (2025). Consumer Wearable Usage to Collect Health Data Among Adults Living in Germany: Nationwide Observational Survey Study. JMIR mHealth and uHealth, 13, e59199. https://doi.org/10.2196/59199

Morokuma, S., Hayashi, T., Kanegae, M., Mizukami, Y., Asano, S., Kimura, I., Tateizumi, Y., Ueno, H.,Ikeda, S., & Niizeki, K. (2023). Deep learning-based sleep stage classification with cardiorespiratory and body movement activities in individuals with suspected sleep disorders. Scientific reports, 13(1), 17730. https://doi.org/10.1038/s41598-023-45020-7 https://pmc.ncbi.nlm.nih.gov/articles/PMC10584883/ 

Rundo, J. V., & Downey, R., 3rd (2019). Polysomnography. Handbook of clinical neurology, 160, 381–392 https://doi.org/10.1016/B978-0-444-64032-1.00025-4

Satapathy, S.K., Brahma, B., Panda, B. et al. Machine learning-empowered sleep staging classification using multi-modality signals. BMC Med Inform Decis Mak 24, 119 (2024). https://doi.org/10.1186/s12911-024-02522-2 https://link.springer.com/article/10.1186/s12911-024-02522-2

Sleep Cycle. (n.d.). The Sleep Cycle Advantage. https://sleepcycle.com/partnerships/value-of-sleep-cycle. Accessed 28.04.2026.

Statistisches Bundesamt (2022). Daten aus den Laufenden Wirtschaftsrechnungen (LWR) zur Ausstattung privater Haushalte mit Informationstechnik. https://www.destatis.de/DE/Themen/Gesellschaft-Umwelt/Einkommen-Konsum-Lebensbedingungen/Ausstattung-Gebrauchsgueter/Tabellen/a-infotechnik-d-lwr.html