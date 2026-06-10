**Working on this repo?** See [SETUP.md](SETUP.md) for installation instructions.

**Contributors:** [@Sieberuni](https://github.com/Sieberuni), [@manmago](https://github.com/manmago)

**Looking for detailed project documentation?** See [project.md](project.md).

---
# Sleep Classification and Hypnogram with Machine Learning

In this project, we built an app that uses phone sensors movement data and recognizes different sleep phases.
We started off with binary sleep-wake classification and then expanded the scope to recognize REM, light and deep sleep.

The Streamlit demo uses compact example-night bundles in `data/example_nights/` so the app can load and display demo nights without shipping the raw CSV source folders in GitHub.

Here you will find a short **tutorial** for our app. Please note that using the app locally will lead to better results.

## Tutorial

Open our [Streamlit app](https://ml4b-sleep-classification.streamlit.app/).

*Sleep-phases mode*: You will see a "demo" version of a final sleep phase classification app.   
It provides you with basic sleep information such as sleep duration and duration of different sleep phases. Below you will find different diagrams displaying more information.  
For now, this remains a "demo" version, since no "real-life" data is used to compile the labels. 

*Binary mode*: If you toggle the radio button to the binary mode on the top left, you will see an early experimental version of the app. Since we focussed on binary classification in the beginning, it is a more experimental version which has some features for testing and exploring. 

In the online/clodud Streamlit app, both versions use a GitHub-friendly, 3x compressed .joblib demo night. It was not used for training and serves as validation. However, please note that this reduces the quality of some of the apps features.
It is also not possible to upload your own datasets.

If you want to validate against your own sleep data, please run the app locally. This way you will experience the best results. 



