"""
Utility Functions for Heart Disease Prediction
================================================
Visualization helpers and common utility functions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
import warnings
warnings.filterwarnings('ignore')


# ─── Color Palette ───
COLORS = {
    'primary': '#e53935',
    'secondary': '#ff6b6b',
    'success': '#43e97b',
    'danger': '#ff416c',
    'warning': '#f7971e',
    'info': '#4facfe',
    'disease': '#ff416c',
    'healthy': '#43e97b',
}

PALETTE = ['#e53935', '#ff6b6b', '#f093fb', '#4facfe', '#43e97b', '#f7971e', '#00f2fe', '#764ba2']


# ─────────────────────────────────────────────
#  CONFUSION MATRIX PLOT
# ─────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, model_name='Model', ax=None):
    """Plot a styled confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    sns.heatmap(
        cm, annot=True, fmt='d', cmap='RdPu',
        xticklabels=['No Disease', 'Disease'],
        yticklabels=['No Disease', 'Disease'],
        ax=ax, cbar_kws={'shrink': 0.8},
        annot_kws={'size': 16, 'weight': 'bold'}
    )
    ax.set_title(f'{model_name}\nConfusion Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)

    return ax


# ─────────────────────────────────────────────
#  ROC CURVE PLOT
# ─────────────────────────────────────────────
def plot_roc_curves(results, y_test, ax=None):
    """Plot ROC curves for all models."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for i, (name, data) in enumerate(results.items()):
        if data['y_prob'] is not None:
            fpr, tpr, _ = roc_curve(y_test, data['y_prob'])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)],
                    lw=2, label=f'{name} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — All Models', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    return ax


# ─────────────────────────────────────────────
#  FEATURE IMPORTANCE
# ─────────────────────────────────────────────
def plot_feature_importance(model, feature_names, top_n=15, ax=None):
    """Plot feature importance for tree-based models."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        sorted_idx = np.argsort(importance)[-top_n:]

        ax.barh(
            range(len(sorted_idx)),
            importance[sorted_idx],
            color=PALETTE[:len(sorted_idx)],
            edgecolor='white', linewidth=0.5
        )
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=10)
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
    elif hasattr(model, 'coef_'):
        # Logistic Regression
        coefs = model.coef_[0]
        sorted_idx = np.argsort(np.abs(coefs))[-top_n:]

        colors = ['#43e97b' if c > 0 else '#ff416c' for c in coefs[sorted_idx]]
        ax.barh(
            range(len(sorted_idx)),
            coefs[sorted_idx],
            color=colors,
            edgecolor='white', linewidth=0.5
        )
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=10)
        ax.set_xlabel('Coefficient Value', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Coefficients', fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)

    return ax


# ─────────────────────────────────────────────
#  METRICS COMPARISON BAR CHART
# ─────────────────────────────────────────────
def plot_metrics_comparison(metrics_df, ax=None):
    """Plot grouped bar chart comparing model metrics."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    x = np.arange(len(metrics_df))
    width = 0.15

    for i, metric in enumerate(metric_cols):
        ax.bar(x + i * width, metrics_df[metric], width,
               label=metric, color=PALETTE[i], edgecolor='white', linewidth=0.5)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(metrics_df['Model'], fontsize=10, rotation=15, ha='right')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])

    return ax


# ─────────────────────────────────────────────
#  CLASSIFICATION REPORT TO DATAFRAME
# ─────────────────────────────────────────────
def get_classification_report_df(y_true, y_pred, model_name='Model'):
    """Convert classification report to a tidy DataFrame."""
    report = classification_report(y_true, y_pred, target_names=['No Disease', 'Disease'], output_dict=True)
    df = pd.DataFrame(report).transpose()
    df['Model'] = model_name
    return df
