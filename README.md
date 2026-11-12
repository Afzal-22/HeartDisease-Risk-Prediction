# Heart Disease Predictor

A Streamlit web app that estimates whether a person is at high or low risk of heart disease, using a Logistic Regression model. 

**Live demo:** [Heart Disease Predictor](https://heartdisease-risk-prediction-vr5vihr8donxhrmfbjkxvn.streamlit.app/)

## Features

- Simple form with help tooltips
- High / low risk result with a risk gauge
- Notes for unusual values
- CSV download of the report and of the session history

## Project structure

```
├── app.py
├── Logistic_Regression_heart.pkl
├── scaler.pkl
├── columns.pkl
├── requirements.txt
└── notebook/
    ├── Heartdisease.ipynb
    └── heart.csv
```

## Dataset and model

- `heart.csv`: 918 records, 11 features, target `HeartDisease`
- Preprocessing: zero values in Cholesterol and RestingBP replaced with the mean, one-hot encoding, 80/20 stratified split, StandardScaler
- Best of 5 models: Logistic Regression with 0.8804 accuracy and 0.8922 F1 (KNN, Naive Bayes, Decision Tree and SVM were also tested)

## Deploy

Push the project to GitHub, then create a new app on share.streamlit.io with `app.py` as the main file.

## Note

For learning and screening only. It does not replace a doctor's advice.
