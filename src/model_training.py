"""
Model Training Module for Heart Disease Prediction
=====================================================
Trains 5 ML models, handles class imbalance with SMOTE,
performs hyperparameter tuning, and saves the best model.

Models: Logistic Regression, Random Forest, Decision Tree, SVM, XGBoost
Dataset: Cardiovascular Disease (70,000 records)
"""

import numpy as np
import pandas as pd
import joblib
import os
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

# Note: SMOTE not used for this large balanced dataset — not needed
# Kept as optional for consistency

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[!] XGBoost not available, skipping.")


# ─────────────────────────────────────────────
#  MODEL DEFINITIONS
# ─────────────────────────────────────────────
def get_models():
    """Return dict of model instances to train."""
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=42, solver='lbfgs', n_jobs=-1
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_split=10,
            min_samples_leaf=5, random_state=42, n_jobs=-1
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10, min_samples_split=10, min_samples_leaf=5,
            random_state=42
        ),
        'SVM': SVC(
            kernel='rbf', C=1.0, gamma='scale',
            probability=True, random_state=42
        ),
    }

    if HAS_XGBOOST:
        models['XGBoost'] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric='logloss',
            n_jobs=-1,
        )

    return models


# ─────────────────────────────────────────────
#  HYPERPARAMETER GRIDS
# ─────────────────────────────────────────────
def get_param_grids():
    """Return hyperparameter grids for tuning."""
    grids = {
        'Logistic Regression': {
            'C': [0.1, 1, 10],
        },
        'Random Forest': {
            'n_estimators': [100, 200],
            'max_depth': [8, 12, 15],
        },
        'Decision Tree': {
            'max_depth': [5, 8, 10, 12],
            'min_samples_split': [5, 10, 20],
            'criterion': ['gini', 'entropy'],
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
        },
    }

    if HAS_XGBOOST:
        grids['XGBoost'] = {
            'n_estimators': [100, 200],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1],
        }

    return grids


# ─────────────────────────────────────────────
#  TRAIN ALL MODELS
# ─────────────────────────────────────────────
def train_all_models(X_train, X_test, y_train, y_test, tune=False):
    """
    Train all models and return results.

    For 70K dataset, SVM is slow — we use a subsample for SVM only.
    """
    models = get_models()
    param_grids = get_param_grids()
    results = {}
    all_metrics = []

    for name, model in models.items():
        print(f"\n{'='*55}")
        print(f"  >> Training: {name}")
        print(f"{'='*55}")

        start_time = time.time()

        # SVM is very slow on 70K samples — subsample for training
        if name == 'SVM' and X_train.shape[0] > 10000:
            print(f"  [i] Subsampling to 10,000 for SVM (original: {X_train.shape[0]})")
            rng = np.random.RandomState(42)
            idx = rng.choice(X_train.shape[0], 10000, replace=False)
            X_tr, y_tr = X_train[idx], y_train[idx]
        else:
            X_tr, y_tr = X_train, y_train

        if tune and name in param_grids:
            print(f"  [i] Tuning hyperparameters...")
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            grid = GridSearchCV(
                model, param_grids[name],
                cv=cv, scoring='f1', n_jobs=-1, verbose=0
            )
            grid.fit(X_tr, y_tr)
            best_model = grid.best_estimator_
            print(f"  [+] Best params: {grid.best_params_}")
        else:
            best_model = model
            best_model.fit(X_tr, y_tr)

        train_time = time.time() - start_time

        # Predictions
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None

        # Metrics
        metrics = {
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_prob) if y_prob is not None else 0,
            'Train Time (s)': round(train_time, 2),
        }

        all_metrics.append(metrics)
        results[name] = {
            'model': best_model,
            'metrics': metrics,
            'y_pred': y_pred,
            'y_prob': y_prob,
        }

        print(f"  Time:      {train_time:.2f}s")
        print(f"  Accuracy:  {metrics['Accuracy']:.4f}")
        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"  Recall:    {metrics['Recall']:.4f}")
        print(f"  F1-Score:  {metrics['F1-Score']:.4f}")
        print(f"  ROC-AUC:   {metrics['ROC-AUC']:.4f}")

    metrics_df = pd.DataFrame(all_metrics)

    best_idx = metrics_df['F1-Score'].idxmax()
    best_name = metrics_df.loc[best_idx, 'Model']
    print(f"\n{'#'*55}")
    print(f"  BEST MODEL: {best_name}")
    print(f"     F1-Score: {metrics_df.loc[best_idx, 'F1-Score']:.4f}")
    print(f"     ROC-AUC:  {metrics_df.loc[best_idx, 'ROC-AUC']:.4f}")
    print(f"{'#'*55}")

    return results, metrics_df


# ─────────────────────────────────────────────
#  SAVE / LOAD MODELS
# ─────────────────────────────────────────────
def save_best_model(results, metrics_df, feature_names, scaler, save_dir='models'):
    """Save the best model, scaler, and metadata."""
    os.makedirs(save_dir, exist_ok=True)

    best_idx = metrics_df['F1-Score'].idxmax()
    best_name = metrics_df.loc[best_idx, 'Model']
    best_model = results[best_name]['model']

    joblib.dump(best_model, os.path.join(save_dir, 'best_model.pkl'))
    joblib.dump(scaler, os.path.join(save_dir, 'preprocessor.pkl'))

    metadata = {
        'best_model_name': best_name,
        'feature_names': feature_names,
        'metrics': metrics_df.to_dict('records'),
    }
    joblib.dump(metadata, os.path.join(save_dir, 'model_metadata.pkl'))

    results_to_save = {}
    for name, data in results.items():
        results_to_save[name] = {
            'model': data['model'],
            'y_pred': data['y_pred'],
            'y_prob': data['y_prob'],
        }
    joblib.dump(results_to_save, os.path.join(save_dir, 'all_results.pkl'))

    print(f"\n[+] Saved to '{save_dir}/':")
    print(f"    best_model.pkl ({best_name})")
    print(f"    preprocessor.pkl")
    print(f"    model_metadata.pkl")
    print(f"    all_results.pkl")

    return best_name, best_model


def load_model(model_dir='models'):
    """Load saved model, scaler, and metadata."""
    model = joblib.load(os.path.join(model_dir, 'best_model.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'preprocessor.pkl'))
    metadata = joblib.load(os.path.join(model_dir, 'model_metadata.pkl'))
    return model, scaler, metadata


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.data_preprocessing import full_preprocessing_pipeline

    filepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'heart.csv')
    X_train, X_test, y_train, y_test, features, scaler, df_clean = full_preprocessing_pipeline(filepath)

    results, metrics_df = train_all_models(X_train, X_test, y_train, y_test, tune=False)

    print("\nMetrics Summary:")
    print(metrics_df.to_string(index=False))

    save_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    save_best_model(results, metrics_df, features, scaler, save_dir)
