**Getting started?** See [SETUP.md](SETUP.md) for installation instructions.

**Team:** [@Sieberuni](https://github.com/Sieberuni), [@manmago](https://github.com/manmago)

# 1 Introduction

## Motivation

In Germany, roughly 20% of the population posess a smartwatch or similar device of which majority use it to track health data (Manz et al., 2025).

Why sleep tracking is essential.

However, certain demographic groups, such as older people, people with lower income and lower physical activity levels are less likely to own a wearable device and use it for health tracking (Manz et al., 2025).

Many more posess a phone, yet lack in apps. => Gap => Our application :D  

## Research Questions

1. Can smartphone accelerometer and gyroscope data, collected non-intrusively during nighttime, reliably classify binary sleep states (asleep vs. awake) with at least 75% accuracy?

2. Can we estimate subjective sleep quality and total sleep duration from smartphone sensor patterns, validated against smartwatch sleep tracking data?

3. Can smartphone sensor data, in combination with derived sleep quality metrics, classify sleep phases (light sleep, deep sleep, potentially REM) with meaningful accuracy, and what are the physiological limitations of such classification without direct biometric signals?

- How is this document structured

# 2 Related Work

- What have others done in your area of work/ to answer similar questions?

*Golden standard* of sleep tracking: **Polysomnography (PSG)**. Monitors brain activity, eye and muscle movements, heart rate, oxygen saturation, airflow, and respiratory effort (Rundo & Downey, 2019).

*Wearables* rely on a combination of body movement, electrocardiogram data, blood volume changes, oxygen saturation, microphone data to predict sleep and sleep phases.

*Mobile* applications are more limited in clinical sensor data, therefore lean on audio data or movement measured with accelerometer and gyroscope. Additionally, almost all apps on the market lack empirical evidence, for instance validation against the golden standard, PSG. Apps with such validation studies present weak correlation. One example is the Sleep Cycle application, which claims to use AI-powered analysis technologies (Amanth, 2021; Sleep Cycle, n.d.).

*Most recent GitHub* project from 2025 for wearables by @mkucukos using Logistic Regression, Random Forest, XGBoost on accelerometer and gyroscope data, as well as body temperature, heart rate. => [**Link**](https://github.com/mkucukos/sleep-awake-detection).

There appears to be a gap in reliable and accessible mobile sensor-based sleep tracking. Despite not being able to validate our work against PSG standards, the challenge is to come close to the reliability of wearable devices.

When does one sleep, when is one awake? What are the different sleep phases?

- Discussing existing work in the context of your work

Consequences for our work, processes, models, where we will contribute.

# 3 Methodology

## 3.1 General Methodology

- How did you proceed to achieve your project goals? 

- Describe which steps you have undertaken

- Aim: Others should understand your research process

## 3.2 Data Understanding and Preparation

- Introduce the dataset to the reader

- Describe structure and size of your dataset

- Describe specialities

- Describe how you prepare the dataset for your project

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
&nbsp; (1), 83–86. https://doi.org/10.5935/1984-0063.20200036

Manz, K., Krug, S., Kühnelt, C., Lemcke, J., Öztürk, I., & Loss, J. (2025). Consumer Wearable Usage to  
&nbsp; Collect Health Data Among Adults Living in Germany: Nationwide Observational Survey Study.  
&nbsp; JMIR mHealth and uHealth, 13, e59199. https://doi.org/10.2196/59199

Rundo, J. V., & Downey, R., 3rd (2019). Polysomnography. Handbook of clinical neurology, 160, 381–392
&nbsp; https://doi.org/10.1016/B978-0-444-64032-1.00025-4

Sleep Cycle. (n.d.). The Sleep Cycle Advantage. 
&nbsp; https://sleepcycle.com/partnerships/value-of-sleep-cycle. Accessed 28.04.2026.
