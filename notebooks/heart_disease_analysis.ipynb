{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# ❤️ Heart Disease Prediction Using Patient Health Data\n",
    "\n",
    "---\n",
    "\n",
    "**End-to-End Machine Learning Pipeline**\n",
    "\n",
    "| Detail | Value |\n",
    "|--------|-------|\n",
    "| **Dataset** | Cardiovascular Disease Dataset (Kaggle) |\n",
    "| **Records** | 70,000 patients |\n",
    "| **Features** | 11 clinical parameters + 1 target |\n",
    "| **Models** | Logistic Regression, Random Forest, Decision Tree, SVM, XGBoost |\n",
    "| **Target** | Cardiovascular Disease (binary: 0/1) |\n",
    "| **Author** | Capstone Project — Predictive Analytics |\n",
    "\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Import Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import time\n",
    "import joblib\n",
    "import os\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# Scikit-learn\n",
    "from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.metrics import (\n",
    "    accuracy_score, precision_score, recall_score, f1_score,\n",
    "    roc_auc_score, roc_curve, auc, confusion_matrix, classification_report\n",
    ")\n",
    "\n",
    "# XGBoost\n",
    "try:\n",
    "    from xgboost import XGBClassifier\n",
    "    HAS_XGBOOST = True\n",
    "except ImportError:\n",
    "    HAS_XGBOOST = False\n",
    "    print(\"XGBoost not installed. Skipping XGBoost model.\")\n",
    "\n",
    "# Plot style\n",
    "plt.style.use('seaborn-v0_8-whitegrid')\n",
    "sns.set_palette(['#e53935', '#ff6b6b', '#4facfe', '#43e97b', '#f093fb', '#f7971e'])\n",
    "\n",
    "print(\"All libraries imported successfully!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## 2. Load Dataset\n",
    "\n",
    "The **Cardiovascular Disease Dataset** from Kaggle contains **70,000 patient records** with 11 clinical features and 1 binary target variable.\n",
    "\n",
    "**Source:** https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('../data/heart.csv')\n",
    "print(f\"Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns\")\n",
    "print(f\"\\nColumns: {df.columns.tolist()}\")\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## 3. Data Overview\n",
    "\n",
    "### 3.1 Dataset Shape & Data Types"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(f\"Shape: {df.shape}\")\n",
    "print(f\"\\nData Types:\")\n",
    "df.dtypes"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3.2 Descriptive Statistics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.describe().round(2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3.3 Missing Values & Duplicates"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Missing Values:\")\n",
    "print(df.isnull().sum())\n",
    "print(f\"\\nDuplicated rows: {df.duplicated().sum()}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## 4. Data Cleaning\n",
    "\n",
    "Steps:\n",
    "1. Drop the `id` column (not a feature)\n",
    "2. Convert `age` from days to years\n",
    "3. Remove physiologically impossible blood pressure outliers\n",
    "4. Filter extreme height/weight outliers (1st–99th percentile)\n",
    "5. Remove duplicate records"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Drop ID column\n",
    "if 'id' in df.columns:\n",
    "    df.drop('id', axis=1, inplace=True)\n",
    "    print(\"Dropped 'id' column\")\n",
    "\n",
    "# Convert age from days to years\n",
    "if df['age'].max() > 200:\n",
    "    df['age'] = (df['age'] / 365.25).astype(int)\n",
    "    print(f\"Converted age to years. Range: {df['age'].min()} - {df['age'].max()}\")\n",
    "\n",
    "print(f\"\\nShape before cleaning: {df.shape}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Remove blood pressure outliers\n",
    "# Systolic: 60-250 mmHg, Diastolic: 40-200 mmHg\n",
    "df = df[(df['ap_hi'] >= 60) & (df['ap_hi'] <= 250)]\n",
    "df = df[(df['ap_lo'] >= 40) & (df['ap_lo'] <= 200)]\n",
    "\n",
    "# Systolic must be greater than diastolic\n",
    "df = df[df['ap_hi'] > df['ap_lo']]\n",
    "\n",
    "# Remove extreme height/weight outliers (1st-99th percentile)\n",
    "for col in ['height', 'weight']:\n",
    "    q_low = df[col].quantile(0.01)\n",
    "    q_hi = df[col].quantile(0.99)\n",
    "    df = df[(df[col] >= q_low) & (df[col] <= q_hi)]\n",
    "    print(f\"{col}: kept range [{q_low:.0f}, {q_hi:.0f}]\")\n",
    "\n",
    "# Remove duplicates\n",
    "n_dupes = df.duplicated().sum()\n",
    "df.drop_duplicates(inplace=True)\n",
    "df.reset_index(drop=True, inplace=True)\n",
    "\n",
    "print(f\"\\nRemoved {n_dupes} duplicates\")\n",
    "print(f\"Shape after cleaning: {df.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4.1 Target Distribution"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Target Distribution:\")\n",
    "print(df['cardio'].value_counts())\n",
    "print(f\"\\nDisease Rate: {df['cardio'].mean():.2%}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## 5. Exploratory Data Analysis (EDA)\n",
    "\n",
    "### 5.1 Create Readable Labels"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_eda = df.copy()\n",
    "df_eda['gender_label'] = df_eda['gender'].map({1: 'Female', 2: 'Male'})\n",
    "df_eda['chol_label'] = df_eda['cholesterol'].map({1: 'Normal', 2: 'Above Normal', 3: 'Well Above Normal'})\n",
    "df_eda['gluc_label'] = df_eda['gluc'].map({1: 'Normal', 2: 'Above Normal', 3: 'Well Above Normal'})\n",
    "df_eda['smoke_label'] = df_eda['smoke'].map({1: 'Yes', 0: 'No'})\n",
    "df_eda['alco_label'] = df_eda['alco'].map({1: 'Yes', 0: 'No'})\n",
    "df_eda['active_label'] = df_eda['active'].map({1: 'Yes', 0: 'No'})\n",
    "df_eda['cardio_label'] = df_eda['cardio'].map({1: 'Disease', 0: 'No Disease'})\n",
    "df_eda['bmi'] = df_eda['weight'] / ((df_eda['height'] / 100) ** 2)\n",
    "\n",
    "# Define color scheme\n",
    "colors = ['#43e97b', '#ff416c']\n",
    "palette = {'No Disease': '#43e97b', 'Disease': '#ff416c'}\n",
    "\n",
    "print(\"Labels created.\")\n",
    "df_eda.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5.2 Target Distribution (Donut Chart)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, ax = plt.subplots(1, 1, figsize=(7, 7))\n",
    "wedges, texts, autotexts = ax.pie(\n",
    "    df['cardio'].value_counts().values,\n",
    "    labels=['No Disease', 'Disease'],\n",
    "    colors=colors, autopct='%1.1f%%',\n",
    "    startangle=90, pctdistance=0.85,\n",
    "    wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2),\n",
    "    textprops={'fontsize': 13, 'fontweight': 'bold'}\n",
    ")\n",
    "ax.set_title('Cardiovascular Disease Distribution (70K Patients)', fontsize=16, fontweight='bold', pad=20)\n",
    "plt.tight_layout()\n",
    "plt.savefig('../models/eda_target_distribution.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5.3 Age Distribution by Disease"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "# Histogram\n",
    "for label, color in zip(['No Disease', 'Disease'], colors):\n",
    "    sub = df_eda[df_eda['cardio_label'] == label]\n",
    "    axes[0].hist(sub['age'], bins=25, alpha=0.6, color=color, label=label, edgecolor='white')\n",
    "axes[0].set_xlabel('Age (years)', fontsize=12)\n",
    "axes[0].set_ylabel('Count', fontsize=12)\n",
    "axes[0].set_title('Age Distribution by Disease', fontsize=14, fontweight='bold')\n",
    "axes[0].legend(fontsize=11)\n",
    "\n",
    "# Box plot\n",
    "sns.boxplot(data=df_eda, x='cardio_label', y='age', palette=palette, ax=axes[1])\n",
    "axes[1].set_xlabel('Diagnosis', fontsize=12)\n",
    "axes[1].set_ylabel('Age (years)', fontsize=12)\n",
    "axes[1].set_title('Age by Diagnosis', fontsize=14, fontweight='bold')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig('../models/eda_age.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5.4 Blood Pressure Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "sns.boxplot(data=df_eda, x='cardio_label', y='ap_hi', palette=palette, ax=axes[0])\n",
    "axes[0].set_xlabel('Diagnosis', fontsize=12)\n",
    "axes[0].set_ylabel('Systolic BP (mmHg)', fontsize=12)\n",
    "axes[0].set_title('Systolic Blood Pressure by Disease', fontsize=14, fontweight='bold')\n",
    "\n",
    "sns.boxplot(data=df_eda, x='cardio_label', y='ap_lo', palette=palette, ax=axes[1])\n",
    "axes[1].set_xlabel('Diagnosis', fontsize=12)\n",
    "axes[1].set_ylabel('Diastolic BP (mmHg)', fontsize=12)\n",
    "axes[1].set_title('Diastolic Blood Pressure by Disease', fontsize=14, fontweight='bold')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig('../models/eda_blood_pressure.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5.5 Disease by Gender"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "# Gender count\n",
    "gender_d = df_eda.groupby(['gender_label', 'cardio_label']).size().reset_index(name='Count')\n",
    "sns.barplot(data=gender_d, x='gender_label', y='Count', hue='cardio_label', palette=palette, ax=axes[0])\n",
    "axes[0].set_title('Heart Disease by Gender', fontsize=14, fontweight='bold')\n",
    "axes[0].set_xlabel('')\n",
    "\n",
    "# Disease rate by gender\n",
    "gender_rate = df_eda.groupby('gender_label')['cardio'].mean() * 100\n",
    "gender_rate.plot(kind='bar', color=['#f093fb', '#4facfe'], edgecolor='white', ax=axes[1])\n",
    "axes[1].set_title('Disease Rate by Gender (%)', fontsize=14, fontweight='bold')\n",
    "axes[1].set_xlabel('')\n",
    "axes[1].set_ylabel('Disease Rate (%)')\n",
    "axes[1].tick_params(axis='x', rotation=0)\n",
    "for i, v in enumerate(gender_rate):\n",
    "    axes[1].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=12)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig('../models/eda_gender.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5.6 Cholesterol & BMI Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "# Cholesterol\n",
    "chol_d = df_eda.groupby(['chol_label', 'cardio_label']).size().reset_index(name='Count')\n",
    "sns.barplot(data=chol_d, x='chol_label', y='Count', hue='cardio_label', palette=palette, ax=axes[0])\n",
    "axes[0].set_title('Disease by Cholesterol Level', fontsize=14, fontweight='bold')\n",
    "axes[0].set_xlabel('Cholesterol Level')\n",
    "\n",
    "# BMI\n",
    "sns.boxplot(data=df_eda, x='cardio_label', y='bmi', palette=palette, ax=axes[1])\n",
    "axes[1].set_title('BMI by Diagnosis', fontsize=14, fontweight='bold')\n",
    "axes[1].set_ylabel('BMI (kg/m\\u00b2)')\n",
    "axes[1].set_xlabel('Diagnosis')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig('../models/eda_chol_bmi.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5.7 Lifestyle Factors (Smoking, Alcohol, Physical Activity)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
    "\n",
    "lifestyle_features = [\n",
    "    ('smoke_label', 'Smoking'),\n",
    "    ('alco_label', 'Alcohol Intake'),\n",
    "    ('active_label', 'Physical Activity'),\n",
    "]\n",
    "\n",
    "for i, (col, title) in enumerate(lifestyle_features):\n",
    "    rate = df_eda.groupby(col)['cardio'].mean() * 100\n",
    "    rate.plot(kind='bar', color=['#e53935', '#ff6b6b'], edgecolor='white', ax=axes[i])\n",
    "    axes[i].set_title(f'Disease Rate by {title}', fontsize=14, fontweight='bold')\n",
    "    axes[i].set_ylabel('Disease Rate (%)')\n",
    "    axes[i].set_xlabel('')\n",
    "    axes[i].tick_params(axis='x', rotation=0)\n",
    "    for j, v in enumerate(rate):\n",
    "        axes[i].text(j, v + 0.3, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=11)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig('../models/eda_lifestyle.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5.8 Disease Rate Across All Categorical Features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "cat_features = {\n",
    "    'Gender': 'gender_label',\n",
    "    'Cholesterol': 'chol_label',\n",
    "    'Glucose': 'gluc_label',\n",
    "    'Smoking': 'smoke_label',\n",
    "    'Alcohol': 'alco_label',\n",
    "    'Physical Activity': 'active_label',\n",
    "}\n",
    "\n",
    "rates = []\n",
    "for feat_name, col_name in cat_features.items():\n",
    "    r = df_eda.groupby(col_name)['cardio'].mean() * 100\n",
    "    for cat_val, pct in r.items():\n",
    "        rates.append({'Feature': feat_name, 'Category': str(cat_val), 'Disease Rate (%)': round(pct, 1)})\n",
    "\n",
    "rate_df = pd.DataFrame(rates)\n",
    "rate_pivot = rate_df.pivot(index='Feature', columns='Category', values='Disease Rate (%)')\n",
    "\n",
    "fig, ax = plt.subplots(1, 1, figsize=(12, 5))\n",
    "sns.heatmap(rate_pivot, annot=True, fmt='.1f', cmap='RdYlGn_r', center=50,\n",
    "            linewidths=0.5, ax=ax, cbar_kws={'label': 'Disease Rate (%)'})\n",
    "ax.set_title('Disease Rate (%) Across All Categories', fontsize=16, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.savefig('../models/eda_disease_rate_heatmap.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5.9 Correlation Heatmap"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, ax = plt.subplots(1, 1, figsize=(12, 10))\n",
    "corr = df.corr()\n",
    "mask = np.triu(np.ones_like(corr, dtype=bool))\n",
    "sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',\n",
    "            center=0, linewidths=0.5, ax=ax, square=True,\n",
    "            cbar_kws={'shrink': 0.8})\n",
    "ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)\n",
    "plt.tight_layout()\n",
    "plt.savefig('../models/eda_correlation.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5.10 Correlation with Target (Cardiovascular Disease)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "target_corr = corr['cardio'].drop('cardio').sort_values(ascending=True)\n",
    "\n",
    "fig, ax = plt.subplots(1, 1, figsize=(10, 6))\n",
    "bars = ax.barh(target_corr.index, target_corr.values,\n",
    "               color=['#ff416c' if v > 0 else '#43e97b' for v in target_corr.values],\n",
    "               edgecolor='white', linewidth=0.5)\n",
    "ax.set_xlabel('Correlation Coefficient', fontsize=12)\n",
    "ax.set_title('Correlation with Cardiovascular Disease (Target)', fontsize=14, fontweight='bold')\n",
    "ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)\n",
    "ax.grid(axis='x', alpha=0.3)\n",
    "plt.tight_layout()\n",
    "plt.savefig('../models/eda_target_correlation.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## 6. Feature Engineering\n",
    "\n",
    "Creating derived medical features:\n",
    "\n",
    "| Feature | Formula | Medical Significance |\n",
    "|---------|---------|---------------------|\n",
    "| `bmi` | weight / (height/100)² | Body Mass Index — obesity indicator |\n",
    "| `pulse_pressure` | systolic − diastolic | Arterial stiffness |\n",
    "| `map_bp` | diastolic + ⅓(systolic − diastolic) | Mean Arterial Pressure |\n",
    "| `bp_ratio` | systolic / diastolic | Blood pressure ratio |\n",
    "| `age_bmi` | age × bmi | Age-obesity interaction |"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# BMI = weight(kg) / height(m)^2\n",
    "df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)\n",
    "\n",
    "# Pulse Pressure = Systolic - Diastolic\n",
    "df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']\n",
    "\n",
    "# Mean Arterial Pressure = Diastolic + 1/3 * (Systolic - Diastolic)\n",
    "df['map_bp'] = df['ap_lo'] + (df['ap_hi'] - df['ap_lo']) / 3\n",
    "\n",
    "# BP ratio\n",
    "df['bp_ratio'] = df['ap_hi'] / df['ap_lo']\n",
    "\n",
    "# Age * BMI interaction\n",
    "df['age_bmi'] = df['age'] * df['bmi']\n",
    "\n",
    "print(f\"Shape after feature engineering: {df.shape}\")\n",
    "print(f\"New features: bmi, pulse_pressure, map_bp, bp_ratio, age_bmi\")\n",
    "df[['bmi', 'pulse_pressure', 'map_bp', 'bp_ratio', 'age_bmi']].describe().round(2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## 7. Data Preprocessing\n",
    "\n",
    "### 7.1 Separate Features and Target"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df.drop('cardio', axis=1)\n",
    "y = df['cardio']\n",
    "\n",
    "print(f\"Features shape: {X.shape}\")\n",
    "print(f\"Target shape: {y.shape}\")\n",
    "print(f\"Target distribution:\\n{y.value_counts()}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 7.2 One-Hot Encoding"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# One-hot encode cholesterol and glucose (ordinal categories: 1, 2, 3)\n",
    "cat_cols = ['cholesterol', 'gluc']\n",
    "X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=float)\n",
    "\n",
    "print(f\"Shape after encoding: {X.shape}\")\n",
    "print(f\"Columns: {X.columns.tolist()}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 7.3 Feature Scaling (StandardScaler)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "num_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo',\n",
    "            'bmi', 'pulse_pressure', 'map_bp', 'bp_ratio', 'age_bmi']\n",
    "\n",
    "scaler = StandardScaler()\n",
    "X[num_cols] = scaler.fit_transform(X[num_cols])\n",
    "X = X.astype(float)\n",
    "feature_names = X.columns.tolist()\n",
    "\n",
    "print(f\"Scaled {len(num_cols)} numerical features\")\n",
    "print(f\"Total features: {len(feature_names)}\")\n",
    "X.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 7.4 Train-Test Split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X.values, y.values, test_size=0.2, random_state=42, stratify=y\n",
    ")\n",
    "\n",
    "print(f\"Train set: {X_train.shape[0]:,} samples\")\n",
    "print(f\"Test set:  {X_test.shape[0]:,} samples\")\n",
    "print(f\"Disease rate (train): {y_train.mean():.2%}\")\n",
    "print(f\"Disease rate (test):  {y_test.mean():.2%}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## 8. Model Training\n",
    "\n",
    "Training **5 classification models** on the cardiovascular disease dataset:\n",
    "\n",
    "| # | Model | Type |\n",
    "|---|-------|------|\n",
    "| 1 | Logistic Regression | Linear |\n",
    "| 2 | Random Forest | Ensemble (Bagging) |\n",
    "| 3 | Decision Tree | Tree-based |\n",
    "| 4 | SVM | Kernel-based |\n",
    "| 5 | XGBoost | Ensemble (Boosting) |\n",
    "\n",
    "> **Note:** SVM is subsampled to 10,000 training points (too slow on 50K+ samples)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define models\n",
    "models = {\n",
    "    'Logistic Regression': LogisticRegression(\n",
    "        max_iter=1000, random_state=42, solver='lbfgs', n_jobs=-1\n",
    "    ),\n",
    "    'Random Forest': RandomForestClassifier(\n",
    "        n_estimators=200, max_depth=12, min_samples_split=10,\n",
    "        min_samples_leaf=5, random_state=42, n_jobs=-1\n",
    "    ),\n",
    "    'Decision Tree': DecisionTreeClassifier(\n",
    "        max_depth=10, min_samples_split=10, min_samples_leaf=5, random_state=42\n",
    "    ),\n",
    "    'SVM': SVC(\n",
    "        kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42\n",
    "    ),\n",
    "}\n",
    "\n",
    "if HAS_XGBOOST:\n",
    "    models['XGBoost'] = XGBClassifier(\n",
    "        n_estimators=200, max_depth=6, learning_rate=0.1,\n",
    "        subsample=0.8, colsample_bytree=0.8,\n",
    "        random_state=42, eval_metric='logloss', n_jobs=-1\n",
    "    )\n",
    "\n",
    "print(f\"Models to train: {list(models.keys())}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "results = {}\n",
    "all_metrics = []\n",
    "\n",
    "for name, model in models.items():\n",
    "    print(f\"\\n{'='*55}\")\n",
    "    print(f\"  >> Training: {name}\")\n",
    "    print(f\"{'='*55}\")\n",
    "\n",
    "    # SVM subsample (too slow on 50K+ samples)\n",
    "    if name == 'SVM' and X_train.shape[0] > 10000:\n",
    "        print(f\"  Subsampling to 10,000 for SVM (original: {X_train.shape[0]:,})\")\n",
    "        rng = np.random.RandomState(42)\n",
    "        idx = rng.choice(X_train.shape[0], 10000, replace=False)\n",
    "        X_tr, y_tr = X_train[idx], y_train[idx]\n",
    "    else:\n",
    "        X_tr, y_tr = X_train, y_train\n",
    "\n",
    "    start = time.time()\n",
    "    model.fit(X_tr, y_tr)\n",
    "    train_time = time.time() - start\n",
    "\n",
    "    # Predictions\n",
    "    y_pred = model.predict(X_test)\n",
    "    y_prob = model.predict_proba(X_test)[:, 1]\n",
    "\n",
    "    # Metrics\n",
    "    metrics = {\n",
    "        'Model': name,\n",
    "        'Accuracy': accuracy_score(y_test, y_pred),\n",
    "        'Precision': precision_score(y_test, y_pred),\n",
    "        'Recall': recall_score(y_test, y_pred),\n",
    "        'F1-Score': f1_score(y_test, y_pred),\n",
    "        'ROC-AUC': roc_auc_score(y_test, y_prob),\n",
    "        'Train Time (s)': round(train_time, 2),\n",
    "    }\n",
    "    all_metrics.append(metrics)\n",
    "    results[name] = {'model': model, 'y_pred': y_pred, 'y_prob': y_prob, 'metrics': metrics}\n",
    "\n",
    "    print(f\"  Time: {train_time:.2f}s\")\n",
    "    print(f\"  Accuracy:  {metrics['Accuracy']:.4f}\")\n",
    "    print(f\"  Precision: {metrics['Precision']:.4f}\")\n",
    "    print(f\"  Recall:    {metrics['Recall']:.4f}\")\n",
    "    print(f\"  F1-Score:  {metrics['F1-Score']:.4f}\")\n",
    "    print(f\"  ROC-AUC:   {metrics['ROC-AUC']:.4f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 8.1 Classification Reports"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "for name, data in results.items():\n",
    "    print(f\"\\n{'='*50}\")\n",
    "    print(f\"  Classification Report: {name}\")\n",
    "    print(f\"{'='*50}\")\n",
    "    print(classification_report(y_test, data['y_pred'], target_names=['No Disease', 'Disease']))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 8.2 Metrics Comparison Table"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "metrics_df = pd.DataFrame(all_metrics)\n",
    "\n",
    "# Highlight best model\n",
    "best_idx = metrics_df['F1-Score'].idxmax()\n",
    "best_name = metrics_df.loc[best_idx, 'Model']\n",
    "\n",
    "print(f\"\\n{'#'*50}\")\n",
    "print(f\"  BEST MODEL: {best_name}\")\n",
    "print(f\"  F1-Score: {metrics_df.loc[best_idx, 'F1-Score']:.4f}\")\n",
    "print(f\"  ROC-AUC:  {metrics_df.loc[best_idx, 'ROC-AUC']:.4f}\")\n",
    "print(f\"{'#'*50}\")\n",
    "\n",
    "metrics_df.style.highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],\n",
    "                                color='#43e97b')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## 9. Model Evaluation\n",
    "\n",
    "### 9.1 Performance Metrics Comparison"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, ax = plt.subplots(1, 1, figsize=(14, 6))\n",
    "\n",
    "metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']\n",
    "x = np.arange(len(metrics_df))\n",
    "width = 0.15\n",
    "bar_colors = ['#e53935', '#ff6b6b', '#f093fb', '#4facfe', '#43e97b']\n",
    "\n",
    "for i, metric in enumerate(metric_cols):\n",
    "    bars = ax.bar(x + i * width, metrics_df[metric], width,\n",
    "                  label=metric, color=bar_colors[i], edgecolor='white', linewidth=0.5)\n",
    "\n",
    "ax.set_xticks(x + width * 2)\n",
    "ax.set_xticklabels(metrics_df['Model'], fontsize=11, rotation=15, ha='right')\n",
    "ax.set_ylabel('Score', fontsize=12)\n",
    "ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')\n",
    "ax.legend(loc='lower right', fontsize=10)\n",
    "ax.grid(True, axis='y', alpha=0.3)\n",
    "ax.set_ylim([0, 1.1])\n",
    "plt.tight_layout()\n",
    "plt.savefig('../models/eval_metrics_comparison.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 9.2 ROC Curves"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, ax = plt.subplots(1, 1, figsize=(9, 7))\n",
    "model_colors = ['#e53935', '#ff6b6b', '#f093fb', '#4facfe', '#43e97b']\n",
    "\n",
    "for i, (name, data) in enumerate(results.items()):\n",
    "    fpr, tpr, _ = roc_curve(y_test, data['y_prob'])\n",
    "    roc_auc_val = auc(fpr, tpr)\n",
    "    ax.plot(fpr, tpr, color=model_colors[i], lw=2.5,\n",
    "            label=f'{name} (AUC = {roc_auc_val:.3f})')\n",
    "\n",
    "ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUC = 0.500)')\n",
    "ax.set_xlim([0, 1])\n",
    "ax.set_ylim([0, 1.05])\n",
    "ax.set_xlabel('False Positive Rate', fontsize=12)\n",
    "ax.set_ylabel('True Positive Rate', fontsize=12)\n",
    "ax.set_title('ROC Curves \\u2014 All Models', fontsize=16, fontweight='bold')\n",
    "ax.legend(loc='lower right', fontsize=10)\n",
    "ax.grid(True, alpha=0.3)\n",
    "plt.tight_layout()\n",
    "plt.savefig('../models/eval_roc_curves.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 9.3 Confusion Matrices"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "n_models = len(results)\n",
    "fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))\n",
    "if n_models == 1:\n",
    "    axes = [axes]\n",
    "\n",
    "for i, (name, data) in enumerate(results.items()):\n",
    "    cm = confusion_matrix(y_test, data['y_pred'])\n",
    "    sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu',\n",
    "                xticklabels=['No Disease', 'Disease'],\n",
    "                yticklabels=['No Disease', 'Disease'],\n",
    "                ax=axes[i], cbar=False,\n",
    "                annot_kws={'size': 14, 'weight': 'bold'})\n",
    "    axes[i].set_title(name, fontsize=13, fontweight='bold')\n",
    "    axes[i].set_xlabel('Predicted')\n",
    "    axes[i].set_ylabel('Actual')\n",
    "\n",
    "plt.suptitle('Confusion Matrices', fontsize=16, fontweight='bold', y=1.02)\n",
    "plt.tight_layout()\n",
    "plt.savefig('../models/eval_confusion_matrices.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 9.4 Feature Importance (Best Model)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "best_model = results[best_name]['model']\n",
    "\n",
    "if hasattr(best_model, 'feature_importances_'):\n",
    "    importance = best_model.feature_importances_\n",
    "    sorted_idx = np.argsort(importance)\n",
    "\n",
    "    fig, ax = plt.subplots(1, 1, figsize=(10, 8))\n",
    "    ax.barh(range(len(sorted_idx)), importance[sorted_idx],\n",
    "            color=['#e53935' if imp > np.mean(importance) else '#ffcdd2'\n",
    "                   for imp in importance[sorted_idx]],\n",
    "            edgecolor='white', linewidth=0.5)\n",
    "    ax.set_yticks(range(len(sorted_idx)))\n",
    "    ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=10)\n",
    "    ax.set_xlabel('Feature Importance', fontsize=12)\n",
    "    ax.set_title(f'Feature Importance \\u2014 {best_name}', fontsize=16, fontweight='bold')\n",
    "    ax.grid(axis='x', alpha=0.3)\n",
    "    plt.tight_layout()\n",
    "    plt.savefig('../models/eval_feature_importance.png', dpi=150, bbox_inches='tight')\n",
    "    plt.show()\n",
    "    \n",
    "    # Top 5 features\n",
    "    print(f\"\\nTop 5 Most Important Features ({best_name}):\")\n",
    "    top5 = sorted_idx[-5:][::-1]\n",
    "    for rank, idx in enumerate(top5, 1):\n",
    "        print(f\"  {rank}. {feature_names[idx]}: {importance[idx]:.4f}\")\n",
    "\n",
    "elif hasattr(best_model, 'coef_'):\n",
    "    coefs = best_model.coef_[0]\n",
    "    sorted_idx = np.argsort(np.abs(coefs))\n",
    "    \n",
    "    fig, ax = plt.subplots(1, 1, figsize=(10, 8))\n",
    "    colors_bar = ['#43e97b' if c > 0 else '#ff416c' for c in coefs[sorted_idx]]\n",
    "    ax.barh(range(len(sorted_idx)), coefs[sorted_idx], color=colors_bar,\n",
    "            edgecolor='white', linewidth=0.5)\n",
    "    ax.set_yticks(range(len(sorted_idx)))\n",
    "    ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=10)\n",
    "    ax.set_xlabel('Coefficient Value', fontsize=12)\n",
    "    ax.set_title(f'Feature Coefficients \\u2014 {best_name}', fontsize=16, fontweight='bold')\n",
    "    ax.grid(axis='x', alpha=0.3)\n",
    "    plt.tight_layout()\n",
    "    plt.savefig('../models/eval_feature_importance.png', dpi=150, bbox_inches='tight')\n",
    "    plt.show()\n",
    "else:\n",
    "    print(f\"Feature importance not available for {best_name}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## 10. Cross Validation (5-Fold)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
    "cv_results = []\n",
    "\n",
    "print(\"5-Fold Cross Validation Results:\")\n",
    "print(f\"{'Model':<25s} | {'Mean F1':>8s} | {'Std F1':>8s}\")\n",
    "print(\"-\" * 50)\n",
    "\n",
    "for name, data in results.items():\n",
    "    model_obj = data['model']\n",
    "    \n",
    "    # Use subsample for SVM CV too\n",
    "    if name == 'SVM' and X.shape[0] > 15000:\n",
    "        rng = np.random.RandomState(42)\n",
    "        idx = rng.choice(X.shape[0], 15000, replace=False)\n",
    "        X_cv, y_cv = X.values[idx], y.values[idx]\n",
    "    else:\n",
    "        X_cv, y_cv = X.values, y.values\n",
    "    \n",
    "    scores = cross_val_score(model_obj, X_cv, y_cv, cv=cv, scoring='f1')\n",
    "    cv_results.append({'Model': name, 'CV Mean F1': scores.mean(), 'CV Std F1': scores.std()})\n",
    "    print(f\"  {name:<23s} | {scores.mean():>8.4f} | {scores.std():>8.4f}\")\n",
    "\n",
    "cv_df = pd.DataFrame(cv_results)\n",
    "cv_df"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## 11. Save Models\n",
    "\n",
    "Saving the best model, scaler, and metadata for the Streamlit dashboard."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "save_dir = '../models'\n",
    "os.makedirs(save_dir, exist_ok=True)\n",
    "\n",
    "# Save best model\n",
    "joblib.dump(results[best_name]['model'], os.path.join(save_dir, 'best_model.pkl'))\n",
    "print(f\"Saved best_model.pkl ({best_name})\")\n",
    "\n",
    "# Save scaler\n",
    "joblib.dump(scaler, os.path.join(save_dir, 'preprocessor.pkl'))\n",
    "print(\"Saved preprocessor.pkl\")\n",
    "\n",
    "# Save metadata\n",
    "metadata = {\n",
    "    'best_model_name': best_name,\n",
    "    'feature_names': feature_names,\n",
    "    'metrics': metrics_df.to_dict('records'),\n",
    "}\n",
    "joblib.dump(metadata, os.path.join(save_dir, 'model_metadata.pkl'))\n",
    "print(\"Saved model_metadata.pkl\")\n",
    "\n",
    "# Save all results\n",
    "results_to_save = {\n",
    "    name: {'model': data['model'], 'y_pred': data['y_pred'], 'y_prob': data['y_prob']}\n",
    "    for name, data in results.items()\n",
    "}\n",
    "joblib.dump(results_to_save, os.path.join(save_dir, 'all_results.pkl'))\n",
    "print(\"Saved all_results.pkl\")\n",
    "\n",
    "print(f\"\\nAll models saved to {save_dir}/\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## 12. Summary\n",
    "\n",
    "### Key Results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(f\"\"\"\n",
    "{'='*60}\n",
    "  PROJECT SUMMARY\n",
    "{'='*60}\n",
    "\n",
    "Dataset:     Cardiovascular Disease (Kaggle)\n",
    "Records:     {len(df):,} patients (after cleaning)\n",
    "Features:    {len(feature_names)} (after engineering + encoding)\n",
    "Models:      {', '.join(results.keys())}\n",
    "Best Model:  {best_name}\n",
    "Best F1:     {metrics_df.loc[best_idx, 'F1-Score']:.4f}\n",
    "Best AUC:    {metrics_df.loc[best_idx, 'ROC-AUC']:.4f}\n",
    "\n",
    "Key Findings:\n",
    "  - Systolic blood pressure is the strongest predictor\n",
    "  - Higher BMI correlates with elevated disease risk\n",
    "  - Cholesterol 'Well Above Normal' nearly doubles disease rates\n",
    "  - Age is a significant risk factor\n",
    "  - Physical activity provides protective effects\n",
    "\n",
    "Saved Artifacts:\n",
    "  - models/best_model.pkl ({best_name})\n",
    "  - models/preprocessor.pkl (StandardScaler)\n",
    "  - models/model_metadata.pkl\n",
    "  - models/all_results.pkl\n",
    "\n",
    "Next Step:\n",
    "  streamlit run app.py\n",
    "\"\"\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Final metrics table\n",
    "metrics_df.style.highlight_max(\n",
    "    subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],\n",
    "    color='#43e97b'\n",
    ").format({\n",
    "    'Accuracy': '{:.4f}',\n",
    "    'Precision': '{:.4f}',\n",
    "    'Recall': '{:.4f}',\n",
    "    'F1-Score': '{:.4f}',\n",
    "    'ROC-AUC': '{:.4f}',\n",
    "    'Train Time (s)': '{:.2f}',\n",
    "})"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbformat_minor": 4,
   "pygments_lexer": "ipython3",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
