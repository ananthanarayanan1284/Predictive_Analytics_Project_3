# ❤️ Heart Disease Prediction Using Patient Health Data

A comprehensive end-to-end Machine Learning project that predicts cardiovascular disease in patients using 70,000 clinical health records from the Kaggle Cardiovascular Disease Dataset.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=for-the-badge&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?style=for-the-badge&logo=scikit-learn)

---

## 🎯 Project Overview

Heart disease is the leading cause of death worldwide. This project builds a Machine Learning pipeline to:

- **Analyze** patient health data and identify cardiovascular risk factors
- **Predict** whether a patient has cardiovascular disease
- **Visualize** insights through an interactive Streamlit dashboard
- **Provide** risk assessment and medical recommendations

### 📊 Dataset

- **Source:** [Cardiovascular Disease Dataset (Kaggle)](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
- **Records:** 70,000 patients
- **Features:** 11 clinical features + 1 target
- **Target:** Cardiovascular Disease (1 = Disease, 0 = No Disease) — ~50% balanced

### 📋 Features

| # | Feature | Description | Type |
|---|---------|-------------|------|
| 1 | `age` | Age (originally in days, converted to years) | Numeric |
| 2 | `gender` | 1 = Female, 2 = Male | Categorical |
| 3 | `height` | Height in cm | Numeric |
| 4 | `weight` | Weight in kg | Numeric |
| 5 | `ap_hi` | Systolic blood pressure | Numeric |
| 6 | `ap_lo` | Diastolic blood pressure | Numeric |
| 7 | `cholesterol` | 1: normal, 2: above normal, 3: well above normal | Categorical |
| 8 | `gluc` | Glucose — 1: normal, 2: above normal, 3: well above normal | Categorical |
| 9 | `smoke` | Smoking (0/1) | Binary |
| 10 | `alco` | Alcohol intake (0/1) | Binary |
| 11 | `active` | Physical activity (0/1) | Binary |
| 12 | `cardio` | Cardiovascular disease (0/1) | Target |

---

## 🚀 Live Demo
Access the interactive dashboard here: [**Heart Disease Predictor App**](https://your-app-name.streamlit.app/)

---

## 📊 Project Visuals

### 1. Cardiovascular Disease Distribution
![Target Distribution](models/eda_target_distribution.png)

### 2. Clinical Risk Factors (Age & BP)
![Age and Blood Pressure Analysis](models/eda_age.png)

### 3. Feature Importance (Best Model)
![Feature Importance](models/eval_feature_importance.png)

### 4. Model Evaluation (ROC Curves)
![ROC Curves](models/eval_roc_curves.png)

---

## 🏗️ Project Structure

```
├── data/
│   └── heart.csv                         # Cardiovascular Disease dataset (70K)
├── notebooks/
│   └── heart_disease_analysis.py         # Full EDA + Modeling script
├── models/
│   ├── best_model.pkl                    # Trained best model
│   ├── preprocessor.pkl                  # Fitted scaler
│   ├── model_metadata.pkl                # Feature names & metrics
│   └── all_results.pkl                   # All model results
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py             # Data cleaning & feature engineering
│   ├── model_training.py                 # Model training pipeline
│   └── utils.py                          # Visualization helpers
├── app.py                                # Streamlit web application
├── requirements.txt                      # Python dependencies
├── README.md                             # Project documentation
└── .gitignore                            # Git ignore rules
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/ananthanarayanan1284/heart-disease-prediction.git
cd heart-disease-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model
```bash
python -m src.model_training
```

### 4. Launch the Dashboard
```bash
streamlit run app.py
```

---

## 🤖 Models Used

| Model | Description |
|-------|-------------|
| Logistic Regression | Baseline linear model |
| Random Forest | Ensemble of decision trees |
| Decision Tree | Single tree classifier with pruning |
| XGBoost | Gradient boosted decision trees |
| SVM | Support vector machine with RBF kernel |

### Feature Engineering
- `bmi` — Body Mass Index (weight / height²)
- `pulse_pressure` — Systolic - Diastolic BP
- `map_bp` — Mean Arterial Pressure
- `bp_ratio` — Systolic / Diastolic ratio
- `age_bmi` — Age × BMI interaction

### Data Cleaning
- Removed physiologically impossible blood pressure values
- Filtered extreme height/weight outliers (1st-99th percentile)
- Converted age from days to years
- Removed duplicate records

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion Matrix, ROC Curve, Feature Importance

---

## 🖥️ Streamlit Dashboard

1. **🏠 Overview** — KPI cards, disease distribution, key insights
2. **📊 Exploratory Analysis** — Interactive charts, correlation heatmaps
3. **❤️ Predict Disease** — Real-time prediction with BMI calculator
4. **📈 Model Performance** — Metrics comparison, ROC curves, confusion matrices

---

## 📈 Key Findings

- **Systolic blood pressure** is the strongest predictor of cardiovascular disease
- **Higher BMI** correlates with elevated disease risk
- **Cholesterol "Well Above Normal"** nearly doubles disease rates
- **Age** — older patients have significantly higher risk
- **Physical activity** provides protective effects
- The dataset is well-balanced (~50/50 split)

---

## 🛠️ Technologies

- **Python 3.8+**
- **Pandas, NumPy** — Data manipulation
- **Scikit-learn** — Model training & evaluation
- **XGBoost** — Gradient boosting
- **Plotly** — Interactive visualizations
- **Streamlit** — Web application framework

---

## 📝 License

This project is developed as a capstone project for Predictive Analytics coursework.

## 👤 Author

**Capstone Project** — Predictive Analytics, Semester 2
