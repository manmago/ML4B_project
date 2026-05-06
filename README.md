**Getting started?** See [SETUP.md](SETUP.md) for installation instructions.

**Team:** [@Sieberuni](https://github.com/Sieberuni), [@manmago](https://github.com/manmago)

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

Sleep when quiet, less movement (Source) What are the different sleep phases?

- Discussing existing work in the context of your work

Consequences for our work, processes, models, where we will contribute.
Related work suggests that Random Forest is the most effectve model, hence it will be considered in testing.

# 3 Methodology

## 3.1 General Methodology

- How did you proceed to achieve your project goals? 

- Describe which steps you have undertaken

- Aim: Others should understand your research process

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

- Describe specialities

Hybrid labeling due to failure of smartwatch to detect wake state successfully. Manual annotations override smartwatch tracking.

### Dataset preparation

1. Data cleaning
- ignore or remove redundant files: TotalAcceleration.csv, AccelerometerUncalibrated.csv, GyroscopeUncalibrated.csv
- missing data
- outliers
- duplicates
- sensor-errors
- smartwatch failure corrected by manual annotation
2. Preprocessing
3. Feature selection
4. Data splitting
5. Potential bias discussion


## 3.3 Modeling and Evaluation

- Describe the model architecture(s) you selected

- Describe how you train your models

- Describe how you evaluate your models/ which metrics you use

# 4 Results

- Describe what artifacts you have build

- Describe the libraries and tools you use

- Describe the concept of your app

- Describe the results you achieve by applying your trained models on unseen data

- Descriptive Language (no judgement, no discussion in this section -> just show what you built)

# 5 Discussion

- Now its time to discuss your results/ artifacts/ app 

- Show the limitations : e.g. missing data, limited training ressources/ GPU availability in Colab, limitaitons of the app

- Discuss your work from an ethics perspective:

- Dangers of the application of your work (for example discrimination through ML models)

- Transparency 

- Effects on society and environment

- Possible sources https://algorithmwatch.org/en/ Have a look at the "Automating Society Report"; https://ainowinstitute.org/ Have a look at this website and their publications

- Further Research: What could be next steps for other researchers (specific research questions)

# 6 Conclusion

- Short summary of your findings and outlook

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

