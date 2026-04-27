"""
Data Preprocessing Module for Heart Disease Prediction
========================================================
Handles data loading, cleaning, feature engineering, encoding, and scaling.

Dataset: Cardiovascular Disease Dataset (70,000 records)
Source: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    """Load the Cardiovascular Disease CSV file."""
    df = pd.read_csv(filepath)
    print(f"[+] Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────
#  CLEAN DATA
# ─────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw dataframe:
    - Drop 'id' column
    - Convert age from days to years
    - Remove physiologically impossible blood pressure outliers
    - Remove extreme height/weight outliers
    - Remove duplicates
    """
    df = df.copy()

    # Drop ID column
    if 'id' in df.columns:
        df.drop('id', axis=1, inplace=True)

    # Convert age from days to years
    df['age'] = (df['age'] / 365.25).astype(int)

    # Remove blood pressure outliers (physiologically impossible)
    # Systolic BP should be 60-250, Diastolic BP 40-200
    df = df[(df['ap_hi'] >= 60) & (df['ap_hi'] <= 250)]
    df = df[(df['ap_lo'] >= 40) & (df['ap_lo'] <= 200)]
    # Systolic should be greater than diastolic
    df = df[df['ap_hi'] > df['ap_lo']]

    # Remove extreme height/weight outliers (below 1st and above 99th percentile)
    for col in ['height', 'weight']:
        q_low = df[col].quantile(0.01)
        q_hi = df[col].quantile(0.99)
        df = df[(df[col] >= q_low) & (df[col] <= q_hi)]

    # Remove duplicates
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        df.drop_duplicates(inplace=True)
        print(f"[+] Removed {n_dupes} duplicate row(s)")

    df.reset_index(drop=True, inplace=True)

    missing = df.isnull().sum().sum()
    print(f"[+] Data cleaned. Shape: {df.shape}. Missing: {missing}")
    return df


# ─────────────────────────────────────────────
#  CREATE READABLE COPY FOR EDA
# ─────────────────────────────────────────────
def make_readable(df: pd.DataFrame) -> pd.DataFrame:
    """Create a human-readable version for EDA."""
    df = df.copy()
    df['gender'] = df['gender'].map({1: 'Female', 2: 'Male'})
    df['cholesterol'] = df['cholesterol'].map({1: 'Normal', 2: 'Above Normal', 3: 'Well Above Normal'})
    df['gluc'] = df['gluc'].map({1: 'Normal', 2: 'Above Normal', 3: 'Well Above Normal'})
    df['smoke'] = df['smoke'].map({1: 'Yes', 0: 'No'})
    df['alco'] = df['alco'].map({1: 'Yes', 0: 'No'})
    df['active'] = df['active'].map({1: 'Yes', 0: 'No'})
    df['cardio'] = df['cardio'].map({1: 'Disease', 0: 'No Disease'})
    return df


# ─────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features:
    - bmi: Body Mass Index (weight / height^2)
    - pulse_pressure: Systolic - Diastolic BP
    - map_bp: Mean Arterial Pressure
    - bp_ratio: Systolic / Diastolic ratio
    - age_bmi: Age * BMI interaction
    """
    df = df.copy()

    # BMI = weight(kg) / height(m)^2
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)

    # Pulse Pressure = Systolic - Diastolic
    df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']

    # Mean Arterial Pressure = Diastolic + 1/3 * (Systolic - Diastolic)
    df['map_bp'] = df['ap_lo'] + (df['ap_hi'] - df['ap_lo']) / 3

    # BP ratio
    df['bp_ratio'] = df['ap_hi'] / df['ap_lo']

    # Age * BMI interaction
    df['age_bmi'] = df['age'] * df['bmi']

    print(f"[+] Feature engineering complete. New shape: {df.shape}")
    return df


# ─────────────────────────────────────────────
#  ENCODE & SCALE
# ─────────────────────────────────────────────
def encode_and_scale(df: pd.DataFrame, fit: bool = True, scaler=None):
    """
    Encode categorical variables and scale numerical features.

    Returns:
        X, y, feature_names, scaler
    """
    df = df.copy()

    # Separate target
    y = df['cardio'].values if 'cardio' in df.columns else None
    if 'cardio' in df.columns:
        df.drop('cardio', axis=1, inplace=True)

    # One-hot encode multi-class categoricals (cholesterol: 1,2,3; gluc: 1,2,3)
    cat_cols = ['cholesterol', 'gluc']
    cat_cols = [c for c in cat_cols if c in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=float)

    # Scale numerical features
    num_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo',
                'bmi', 'pulse_pressure', 'map_bp', 'bp_ratio', 'age_bmi']
    num_cols = [c for c in num_cols if c in df.columns]

    if fit:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        df[num_cols] = scaler.transform(df[num_cols])

    # Ensure all columns are float
    df = df.astype(float)

    feature_names = df.columns.tolist()
    X = df.values

    print(f"[+] Encoding complete. Feature matrix shape: {X.shape}")
    return X, y, feature_names, scaler


# ─────────────────────────────────────────────
#  FULL PIPELINE
# ─────────────────────────────────────────────
def full_preprocessing_pipeline(filepath: str, test_size: float = 0.2, random_state: int = 42):
    """
    Run the complete preprocessing pipeline:
    Load -> Clean -> Engineer -> Encode -> Split

    Returns:
        X_train, X_test, y_train, y_test, feature_names, scaler, df_clean
    """
    df = load_data(filepath)
    df = clean_data(df)
    df_clean = df.copy()

    df = engineer_features(df)
    X, y, feature_names, scaler = encode_and_scale(df, fit=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"\n[i] Train set: {X_train.shape[0]} samples")
    print(f"[i] Test set:  {X_test.shape[0]} samples")
    print(f"[i] Disease rate (train): {y_train.mean():.2%}")
    print(f"[i] Disease rate (test):  {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test, feature_names, scaler, df_clean


# ─────────────────────────────────────────────
#  PREPROCESS SINGLE INPUT (for Streamlit)
# ─────────────────────────────────────────────
def preprocess_single_input(input_dict: dict, scaler, training_columns: list) -> np.ndarray:
    """
    Preprocess a single patient input for prediction.
    """
    df = pd.DataFrame([input_dict])

    # Feature engineering (same as training)
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
    df['map_bp'] = df['ap_lo'] + (df['ap_hi'] - df['ap_lo']) / 3
    df['bp_ratio'] = df['ap_hi'] / df['ap_lo']
    df['age_bmi'] = df['age'] * df['bmi']

    # Drop target if present
    if 'cardio' in df.columns:
        df.drop('cardio', axis=1, inplace=True)

    # One-hot encode
    cat_cols = ['cholesterol', 'gluc']
    cat_cols = [c for c in cat_cols if c in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=float)

    # Align columns
    for col in training_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[training_columns]

    # Scale
    num_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo',
                'bmi', 'pulse_pressure', 'map_bp', 'bp_ratio', 'age_bmi']
    num_cols = [c for c in num_cols if c in df.columns]
    df[num_cols] = scaler.transform(df[num_cols])

    return df.values.astype(float)


if __name__ == '__main__':
    filepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'heart.csv')
    X_train, X_test, y_train, y_test, features, scaler, df_clean = full_preprocessing_pipeline(filepath)
    print(f"\n[+] Pipeline test passed! Features: {len(features)}")
