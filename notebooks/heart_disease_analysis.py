"""
Heart Disease Prediction — Complete Analysis (70K Dataset)
============================================================
End-to-end EDA + Model Training + Evaluation

Dataset: Cardiovascular Disease Dataset (Kaggle) — 70,000 records
Features: 11 clinical parameters + 1 target (cardio)
Models: Logistic Regression, Random Forest, Decision Tree, SVM, XGBoost

Author: Capstone Project — Predictive Analytics
"""

# ===================================================================
#  1. IMPORTS
# ===================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc, confusion_matrix, classification_report
)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette(['#e53935', '#ff6b6b', '#4facfe', '#43e97b', '#f093fb', '#f7971e'])
print("[+] All libraries imported successfully!")


# ===================================================================
#  2. LOAD DATA
# ===================================================================
df = pd.read_csv('../data/heart.csv')
print(f"\n[i] Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Columns: {df.columns.tolist()}")
print(df.head())


# ===================================================================
#  3. DATA OVERVIEW
# ===================================================================
print("\n" + "="*60)
print("  DATA OVERVIEW")
print("="*60)

print(f"\nShape: {df.shape}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nDescriptive Statistics:\n{df.describe().round(2)}")
print(f"\nDuplicated rows: {df.duplicated().sum()}")

# Drop id column
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

# Convert age from days to years
if df['age'].max() > 200:
    df['age'] = (df['age'] / 365.25).astype(int)

# Clean outliers
df = df[(df['ap_hi'] >= 60) & (df['ap_hi'] <= 250)]
df = df[(df['ap_lo'] >= 40) & (df['ap_lo'] <= 200)]
df = df[df['ap_hi'] > df['ap_lo']]
for col in ['height', 'weight']:
    q_low, q_hi = df[col].quantile(0.01), df[col].quantile(0.99)
    df = df[(df[col] >= q_low) & (df[col] <= q_hi)]
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"\n[+] After cleaning: {df.shape}")
print(f"Target distribution:\n{df['cardio'].value_counts()}")
print(f"Disease Rate: {df['cardio'].mean():.2%}")


# ===================================================================
#  4. EDA
# ===================================================================
print("\n" + "="*60)
print("  EXPLORATORY DATA ANALYSIS")
print("="*60)

df_eda = df.copy()
df_eda['gender_label'] = df_eda['gender'].map({1: 'Female', 2: 'Male'})
df_eda['chol_label'] = df_eda['cholesterol'].map({1: 'Normal', 2: 'Above Normal', 3: 'Well Above Normal'})
df_eda['cardio_label'] = df_eda['cardio'].map({1: 'Disease', 0: 'No Disease'})
df_eda['bmi'] = df_eda['weight'] / ((df_eda['height'] / 100) ** 2)

# 4.1 Target Distribution
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
colors = ['#43e97b', '#ff416c']
ax.pie(df['cardio'].value_counts().values, labels=['No Disease', 'Disease'],
       colors=colors, autopct='%1.1f%%', startangle=90, pctdistance=0.85,
       wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2),
       textprops={'fontsize': 12, 'fontweight': 'bold'})
ax.set_title('Cardiovascular Disease Distribution', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('../models/eda_target_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# 4.2 Age Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for label, color in zip(['No Disease', 'Disease'], colors):
    sub = df_eda[df_eda['cardio_label'] == label]
    axes[0].hist(sub['age'], bins=25, alpha=0.6, color=color, label=label, edgecolor='white')
axes[0].set_xlabel('Age (years)')
axes[0].set_title('Age Distribution by Disease', fontsize=14, fontweight='bold')
axes[0].legend()
sns.boxplot(data=df_eda, x='cardio_label', y='age', palette={'No Disease': '#43e97b', 'Disease': '#ff416c'}, ax=axes[1])
axes[1].set_title('Age by Diagnosis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../models/eda_age.png', dpi=150, bbox_inches='tight')
plt.show()

# 4.3 Blood Pressure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(data=df_eda, x='cardio_label', y='ap_hi', palette={'No Disease':'#43e97b','Disease':'#ff416c'}, ax=axes[0])
axes[0].set_title('Systolic BP by Disease', fontsize=14, fontweight='bold')
sns.boxplot(data=df_eda, x='cardio_label', y='ap_lo', palette={'No Disease':'#43e97b','Disease':'#ff416c'}, ax=axes[1])
axes[1].set_title('Diastolic BP by Disease', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../models/eda_blood_pressure.png', dpi=150, bbox_inches='tight')
plt.show()

# 4.4 Cholesterol & BMI
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
chol_d = df_eda.groupby(['chol_label','cardio_label']).size().reset_index(name='Count')
sns.barplot(data=chol_d, x='chol_label', y='Count', hue='cardio_label',
            palette={'No Disease':'#43e97b','Disease':'#ff416c'}, ax=axes[0])
axes[0].set_title('Disease by Cholesterol', fontsize=14, fontweight='bold')
sns.boxplot(data=df_eda, x='cardio_label', y='bmi', palette={'No Disease':'#43e97b','Disease':'#ff416c'}, ax=axes[1])
axes[1].set_title('BMI by Disease', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../models/eda_chol_bmi.png', dpi=150, bbox_inches='tight')
plt.show()

# 4.5 Correlation
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax, square=True)
ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('../models/eda_correlation.png', dpi=150, bbox_inches='tight')
plt.show()


# ===================================================================
#  5. FEATURE ENGINEERING
# ===================================================================
print("\n" + "="*60)
print("  FEATURE ENGINEERING")
print("="*60)

df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
df['map_bp'] = df['ap_lo'] + (df['ap_hi'] - df['ap_lo']) / 3
df['bp_ratio'] = df['ap_hi'] / df['ap_lo']
df['age_bmi'] = df['age'] * df['bmi']
print(f"[+] Shape after engineering: {df.shape}")


# ===================================================================
#  6. PREPROCESSING
# ===================================================================
print("\n" + "="*60)
print("  PREPROCESSING")
print("="*60)

X = df.drop('cardio', axis=1)
y = df['cardio']

cat_cols = ['cholesterol', 'gluc']
X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=float)

num_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi', 'pulse_pressure', 'map_bp', 'bp_ratio', 'age_bmi']
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
X = X.astype(float)
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42, stratify=y)
print(f"[i] Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
print(f"[i] Disease rate (train): {y_train.mean():.2%}, (test): {y_test.mean():.2%}")


# ===================================================================
#  7. MODEL TRAINING
# ===================================================================
print("\n" + "="*60)
print("  MODEL TRAINING")
print("="*60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs', n_jobs=-1),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=10,
                                            min_samples_leaf=5, random_state=42, n_jobs=-1),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, min_samples_split=10, min_samples_leaf=5, random_state=42),
    'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42),
}
if HAS_XGBOOST:
    models['XGBoost'] = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                       subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss', n_jobs=-1)

results = {}
all_metrics = []

for name, model in models.items():
    print(f"\n{'='*55}")
    print(f"  >> Training: {name}")
    print(f"{'='*55}")

    # SVM subsample
    if name == 'SVM' and X_train.shape[0] > 10000:
        print(f"  [i] Subsampling to 10,000 for SVM")
        rng = np.random.RandomState(42)
        idx = rng.choice(X_train.shape[0], 10000, replace=False)
        X_tr, y_tr = X_train[idx], y_train[idx]
    else:
        X_tr, y_tr = X_train, y_train

    start = time.time()
    model.fit(X_tr, y_tr)
    t = time.time() - start

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob),
        'Train Time (s)': round(t, 2),
    }
    all_metrics.append(metrics)
    results[name] = {'model': model, 'y_pred': y_pred, 'y_prob': y_prob, 'metrics': metrics}

    print(f"  Time: {t:.2f}s | Acc: {metrics['Accuracy']:.4f} | F1: {metrics['F1-Score']:.4f} | AUC: {metrics['ROC-AUC']:.4f}")
    print(classification_report(y_test, y_pred, target_names=['No Disease', 'Disease']))

metrics_df = pd.DataFrame(all_metrics)
best_idx = metrics_df['F1-Score'].idxmax()
best_name = metrics_df.loc[best_idx, 'Model']
print(f"\n  BEST MODEL: {best_name} (F1={metrics_df.loc[best_idx,'F1-Score']:.4f})")
print(metrics_df.to_string(index=False))


# ===================================================================
#  8. EVALUATION PLOTS
# ===================================================================
# ROC Curves
fig, ax = plt.subplots(1, 1, figsize=(9, 7))
clrs = ['#e53935', '#ff6b6b', '#f093fb', '#4facfe', '#43e97b']
for i, (n, d) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(y_test, d['y_prob'])
    ax.plot(fpr, tpr, color=clrs[i], lw=2.5, label=f'{n} (AUC={auc(fpr,tpr):.3f})')
ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.5)
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title('ROC Curves — All Models', fontsize=16, fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('../models/eval_roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# Confusion Matrices
n = len(results)
fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
if n == 1: axes = [axes]
for i, (nm, d) in enumerate(results.items()):
    cm = confusion_matrix(y_test, d['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', ax=axes[i], cbar=False,
                xticklabels=['Healthy','Disease'], yticklabels=['Healthy','Disease'],
                annot_kws={'size':14,'weight':'bold'})
    axes[i].set_title(nm, fontsize=13, fontweight='bold')
plt.suptitle('Confusion Matrices', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('../models/eval_confusion.png', dpi=150, bbox_inches='tight')
plt.show()

# Feature Importance
bm = results[best_name]['model']
if hasattr(bm, 'feature_importances_'):
    imp = bm.feature_importances_
    si = np.argsort(imp)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.barh(range(len(si)), imp[si], color=['#e53935' if i>np.mean(imp) else '#ffcdd2' for i in imp[si]])
    ax.set_yticks(range(len(si)))
    ax.set_yticklabels([feature_names[i] for i in si])
    ax.set_title(f'Feature Importance - {best_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../models/eval_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()


# ===================================================================
#  9. SAVE MODELS
# ===================================================================
import joblib, os
save_dir = '../models'
os.makedirs(save_dir, exist_ok=True)

joblib.dump(results[best_name]['model'], os.path.join(save_dir, 'best_model.pkl'))
joblib.dump(scaler, os.path.join(save_dir, 'preprocessor.pkl'))
meta = {'best_model_name': best_name, 'feature_names': feature_names, 'metrics': metrics_df.to_dict('records')}
joblib.dump(meta, os.path.join(save_dir, 'model_metadata.pkl'))
rs = {n: {'model':d['model'],'y_pred':d['y_pred'],'y_prob':d['y_prob']} for n, d in results.items()}
joblib.dump(rs, os.path.join(save_dir, 'all_results.pkl'))

print(f"\n[+] All models saved to {save_dir}/")
print(f"[+] Best model: {best_name}")
print("\nNext step: streamlit run app.py")
