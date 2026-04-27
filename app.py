"""
Heart Disease Prediction — Streamlit Dashboard
====================================================
A premium, interactive web application for predicting
cardiovascular disease in patients using Machine Learning.

Author: Capstone Project — Predictive Analytics
Dataset: Cardiovascular Disease Dataset (70,000 records, Kaggle)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #2d142c 50%, #510a32 100%);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown li,
    [data-testid="stSidebar"] .stMarkdown label {
        color: #e0e0e0 !important;
    }
    .metric-card {
        background: linear-gradient(135deg, #e53935 0%, #ff6b6b 100%);
        border-radius: 16px; padding: 24px; color: white;
        text-align: center; box-shadow: 0 8px 32px rgba(229,57,53,0.25);
        transition: transform 0.3s ease; margin-bottom: 16px;
    }
    .metric-card:hover { transform: translateY(-4px); }
    .metric-card h3 { margin:0; font-size:14px; font-weight:500; opacity:0.9; text-transform:uppercase; letter-spacing:1px; }
    .metric-card h1 { margin:8px 0 0 0; font-size:36px; font-weight:800; }
    .metric-card-green { background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%); box-shadow: 0 8px 32px rgba(0,176,155,0.25); }
    .metric-card-red { background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%); box-shadow: 0 8px 32px rgba(255,65,108,0.25); }
    .metric-card-blue { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); box-shadow: 0 8px 32px rgba(79,172,254,0.25); }
    .metric-card-orange { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); box-shadow: 0 8px 32px rgba(240,147,251,0.25); }
    .prediction-high { background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%); border-radius:16px; padding:32px; color:white; text-align:center; box-shadow: 0 8px 32px rgba(255,65,108,0.3); }
    .prediction-medium { background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%); border-radius:16px; padding:32px; color:#333; text-align:center; box-shadow: 0 8px 32px rgba(247,151,30,0.3); }
    .prediction-low { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); border-radius:16px; padding:32px; color:white; text-align:center; box-shadow: 0 8px 32px rgba(17,153,142,0.3); }
    .section-header { background: linear-gradient(90deg, #e53935, #ff6b6b); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size:28px; font-weight:800; margin-bottom:4px; }
    .info-box { background: linear-gradient(135deg, #ffeef0 0%, #ffd6d6 100%); border-radius:12px; padding:20px; border-left:4px solid #e53935; margin:16px 0; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    .custom-divider { height:3px; background: linear-gradient(90deg, #e53935, #ff6b6b, #f093fb, #4facfe); border:none; border-radius:2px; margin:24px 0; }
</style>
""", unsafe_allow_html=True)


# ─── Constants ───────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'heart.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
TARGET_COLORS = {'No Disease': '#43e97b', 'Disease': '#ff416c'}

GENDER_MAP = {1: 'Female', 2: 'Male'}
CHOL_MAP = {1: 'Normal', 2: 'Above Normal', 3: 'Well Above Normal'}
GLUC_MAP = {1: 'Normal', 2: 'Above Normal', 3: 'Well Above Normal'}
YESNO_MAP = {1: 'Yes', 0: 'No'}


# ─── Data Loading ────────────────────────────────────────
@st.cache_data
def load_raw_data():
    df = pd.read_csv(DATA_PATH)
    if 'id' in df.columns:
        df.drop('id', axis=1, inplace=True)
    # Convert age from days to years if needed
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
    # BMI
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    # Readable labels
    df['gender_label'] = df['gender'].map(GENDER_MAP)
    df['chol_label'] = df['cholesterol'].map(CHOL_MAP)
    df['gluc_label'] = df['gluc'].map(GLUC_MAP)
    df['smoke_label'] = df['smoke'].map(YESNO_MAP)
    df['alco_label'] = df['alco'].map(YESNO_MAP)
    df['active_label'] = df['active'].map(YESNO_MAP)
    df['cardio_label'] = df['cardio'].map({1: 'Disease', 0: 'No Disease'})
    return df


@st.cache_resource
def load_trained_model():
    try:
        model = joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))
        scaler = joblib.load(os.path.join(MODELS_DIR, 'preprocessor.pkl'))
        metadata = joblib.load(os.path.join(MODELS_DIR, 'model_metadata.pkl'))
        all_results = joblib.load(os.path.join(MODELS_DIR, 'all_results.pkl'))
        return model, scaler, metadata, all_results
    except FileNotFoundError:
        return None, None, None, None


def metric_card(title, value, card_class="metric-card"):
    return f'<div class="{card_class}"><h3>{title}</h3><h1>{value}</h1></div>'


def make_gauge(value, title="Disease Risk"):
    color = '#43e97b' if value < 0.3 else ('#f7971e' if value < 0.6 else '#ff416c')
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20, 'family': 'Inter'}},
        number={'suffix': '%', 'font': {'size': 40, 'family': 'Inter'}},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color, 'thickness': 0.3},
               'bgcolor': '#f0f2f6', 'borderwidth': 0,
               'steps': [{'range': [0, 30], 'color': 'rgba(67,233,123,0.15)'},
                         {'range': [30, 60], 'color': 'rgba(247,151,30,0.15)'},
                         {'range': [60, 100], 'color': 'rgba(255,65,108,0.15)'}],
               'threshold': {'line': {'color': color, 'width': 4}, 'thickness': 0.8, 'value': value * 100}}
    ))
    fig.update_layout(height=280, margin=dict(l=30, r=30, t=60, b=20), paper_bgcolor='rgba(0,0,0,0)', font={'family': 'Inter'})
    return fig


# ─── Sidebar ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("# ❤️ Heart Disease Predictor")
    st.markdown("---")
    page = st.radio("**Navigation**",
        ["🏠 Overview", "📊 Exploratory Analysis", "❤️ Predict Disease", "📈 Model Performance"],
        label_visibility="collapsed")
    st.markdown("---")
    st.markdown("<div style='text-align:center;opacity:0.7;font-size:12px;'>"
                "<p>Built with ❤️ using Streamlit</p>"
                "<p>Dataset: Cardiovascular Disease (70K)</p>"
                "<p>© 2026 Capstone Project</p></div>", unsafe_allow_html=True)

df = load_raw_data()
model, scaler, metadata, all_results = load_trained_model()


# ═══════════════════════════════════════════════════════════
#  PAGE 1: OVERVIEW
# ═══════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<p class="section-header">❤️ Cardiovascular Disease Prediction Dashboard</p>', unsafe_allow_html=True)
    st.markdown("Predicting heart disease in patients using **70,000 clinical records** — *powered by ML*")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    total = len(df)
    diseased = df['cardio'].sum()
    healthy = total - diseased
    disease_rate = diseased / total * 100

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(metric_card("Total Patients", f"{total:,}"), unsafe_allow_html=True)
    with c2: st.markdown(metric_card("Heart Disease", f"{diseased:,}", "metric-card metric-card-red"), unsafe_allow_html=True)
    with c3: st.markdown(metric_card("Healthy", f"{healthy:,}", "metric-card metric-card-green"), unsafe_allow_html=True)
    with c4: st.markdown(metric_card("Disease Rate", f"{disease_rate:.1f}%", "metric-card metric-card-orange"), unsafe_allow_html=True)

    st.markdown("")
    c1, c2 = st.columns(2)
    with c1:
        tc = df['cardio_label'].value_counts()
        fig = go.Figure(data=[go.Pie(labels=tc.index, values=tc.values, hole=0.55,
            marker_colors=[TARGET_COLORS.get(l,'#ccc') for l in tc.index],
            textfont_size=14, textinfo='label+percent')])
        fig.update_layout(title=dict(text='Disease Distribution', font=dict(size=18, family='Inter')),
            height=400, paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter'),
            annotations=[dict(text=f'<b>{disease_rate:.1f}%</b><br>Disease', x=0.5, y=0.5, font_size=16, showarrow=False)])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        chol_d = df.groupby(['chol_label','cardio_label']).size().reset_index(name='Count')
        fig = px.bar(chol_d, x='chol_label', y='Count', color='cardio_label',
            barmode='group', color_discrete_map=TARGET_COLORS, title='Disease by Cholesterol Level')
        fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter'),
            xaxis_title='Cholesterol', yaxis_title='Patients', legend_title='Diagnosis')
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for lbl, clr in TARGET_COLORS.items():
            sub = df[df['cardio_label']==lbl]
            fig.add_trace(go.Histogram(x=sub['age'], name=lbl, marker_color=clr, opacity=0.7, nbinsx=30))
        fig.update_layout(title=dict(text='Age Distribution by Disease', font=dict(size=18)),
            barmode='overlay', height=380, xaxis_title='Age (years)', yaxis_title='Count',
            paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter'))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        for lbl, clr in TARGET_COLORS.items():
            sub = df[df['cardio_label']==lbl]
            fig.add_trace(go.Box(y=sub['ap_hi'], name=lbl, marker_color=clr, boxmean='sd'))
        fig.update_layout(title=dict(text='Systolic BP by Diagnosis', font=dict(size=18)),
            height=380, yaxis_title='Systolic BP (mmHg)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter'))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 💡 Key Medical Insights")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="info-box"><b>🩺 Blood Pressure</b><br>'
            'Higher <b>systolic blood pressure</b> is the strongest predictor of cardiovascular disease in this dataset.</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="info-box"><b>⚖️ BMI & Weight</b><br>'
            'Patients with higher <b>BMI and weight</b> show significantly elevated risk of heart disease.</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="info-box"><b>🧪 Cholesterol</b><br>'
            'Patients with <b>"Well Above Normal"</b> cholesterol have nearly <b>2x higher</b> disease rates.</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  PAGE 2: EXPLORATORY ANALYSIS
# ═══════════════════════════════════════════════════════════
elif page == "📊 Exploratory Analysis":
    st.markdown('<p class="section-header">📊 Exploratory Data Analysis</p>', unsafe_allow_html=True)
    st.markdown("Deep dive into **70,000 patient records** — clinical features and risk factors")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Overview", "📊 Categorical", "📈 Numerical", "🔥 Correlation"])

    with tab1:
        c1, c2 = st.columns([1,1])
        with c1:
            st.markdown("#### 📋 Dataset Shape")
            st.info(f"**Rows:** {df.shape[0]:,}  |  **Columns:** {df.shape[1]}")
            orig = ['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active','cardio']
            d = pd.DataFrame({'Column': orig, 'Type': [str(df[c].dtype) for c in orig], 'Unique': [df[c].nunique() for c in orig]})
            st.dataframe(d, use_container_width=True, hide_index=True)
            missing = df[orig].isnull().sum()
            missing = missing[missing > 0]
            if len(missing) == 0: st.success("No missing values!")
            else: st.write(missing)
        with c2:
            st.markdown("#### 📈 Descriptive Statistics")
            st.dataframe(df[orig].describe().round(2), use_container_width=True)
        st.markdown("#### 🔍 Sample Data (first 10 rows)")
        st.dataframe(df[orig].head(10), use_container_width=True)

    with tab2:
        cat_features = {'Gender': 'gender_label', 'Cholesterol': 'chol_label', 'Glucose': 'gluc_label',
                         'Smoking': 'smoke_label', 'Alcohol': 'alco_label', 'Physical Activity': 'active_label'}
        sel = st.selectbox("Select Feature", list(cat_features.keys()), index=1)
        col = cat_features[sel]
        c1, c2 = st.columns(2)
        with c1:
            cnt = df.groupby([col, 'cardio_label']).size().reset_index(name='Count')
            fig = px.bar(cnt, x=col, y='Count', color='cardio_label', barmode='group',
                color_discrete_map=TARGET_COLORS, title=f'Distribution: {sel}')
            fig.update_layout(height=420, paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter'), legend_title='Diagnosis')
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            rate = df.groupby(col)['cardio'].mean().reset_index()
            rate.columns = [col, 'Disease Rate (%)']
            rate['Disease Rate (%)'] *= 100
            fig = px.bar(rate, x=col, y='Disease Rate (%)', color='Disease Rate (%)',
                color_continuous_scale=['#43e97b','#f7971e','#ff416c'], title=f'Disease Rate by {sel}')
            fig.update_layout(height=420, paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter'), showlegend=False)
            overall = df['cardio'].mean() * 100
            fig.add_hline(y=overall, line_dash="dash", line_color="#e53935",
                annotation_text=f"Overall Avg ({overall:.1f}%)", annotation_position="top right")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 📊 Disease Rate Across All Categories")
        rates = []
        for fn, cn in cat_features.items():
            r = df.groupby(cn)['cardio'].mean() * 100
            for v, pct in r.items():
                rates.append({'Feature': fn, 'Category': str(v), 'Disease Rate (%)': round(pct, 1)})
        rdf = pd.DataFrame(rates)
        rpiv = rdf.pivot(index='Feature', columns='Category', values='Disease Rate (%)')
        fig = px.imshow(rpiv, text_auto='.1f', color_continuous_scale=['#43e97b','#ffd200','#ff416c'],
            title='Disease Rate Heatmap (%)', aspect='auto')
        fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter'))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        num_map = {'Age (years)': 'age', 'Systolic BP': 'ap_hi', 'Diastolic BP': 'ap_lo',
                    'Height (cm)': 'height', 'Weight (kg)': 'weight', 'BMI': 'bmi'}
        sel_num = st.selectbox("Select Feature", list(num_map.keys()), index=1)
        ncol = num_map[sel_num]
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x=ncol, color='cardio_label', color_discrete_map=TARGET_COLORS,
                barmode='overlay', opacity=0.7, nbins=40, title=f'{sel_num} by Disease')
            fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter'), legend_title='Diagnosis')
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.violin(df, x='cardio_label', y=ncol, color='cardio_label',
                color_discrete_map=TARGET_COLORS, box=True, title=f'{sel_num} — Violin Plot')
            fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter'), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 📈 Feature Relationships")
        c1, c2 = st.columns(2)
        with c1: x_ax = st.selectbox("X-Axis", list(num_map.keys()), index=0)
        with c2: y_ax = st.selectbox("Y-Axis", list(num_map.keys()), index=1)
        # Subsample for scatter (too many points)
        df_sample = df.sample(n=min(5000, len(df)), random_state=42)
        fig = px.scatter(df_sample, x=num_map[x_ax], y=num_map[y_ax], color='cardio_label',
            color_discrete_map=TARGET_COLORS, opacity=0.4, title=f'{x_ax} vs {y_ax} (5K sample)')
        fig.update_layout(height=450, paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter'), legend_title='Diagnosis')
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        orig_num = ['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active','bmi','cardio']
        on = [c for c in orig_num if c in df.columns]
        corr = df[on].corr()
        fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
            title='Feature Correlation Matrix', zmin=-1, zmax=1)
        fig.update_layout(height=550, paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter'))
        st.plotly_chart(fig, use_container_width=True)

        tc = corr['cardio'].drop('cardio').sort_values()
        fig = go.Figure(go.Bar(x=tc.values, y=tc.index, orientation='h',
            marker_color=['#ff416c' if v > 0 else '#43e97b' for v in tc.values]))
        fig.update_layout(title='Correlation with Cardiovascular Disease', height=400,
            paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter'), xaxis_title='Correlation')
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════
#  PAGE 3: PREDICT HEART DISEASE
# ═══════════════════════════════════════════════════════════
elif page == "❤️ Predict Disease":
    st.markdown('<p class="section-header">❤️ Heart Disease Prediction</p>', unsafe_allow_html=True)
    st.markdown("Enter patient health parameters to predict cardiovascular disease risk")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    if model is None:
        st.error("No trained model found! Run: `python -m src.model_training`")
        st.stop()

    st.markdown(f"**Active Model:** `{metadata['best_model_name']}`  |  **Dataset:** 70,000 patients")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 👤 Demographics")
        age = st.slider("Age (years)", 20, 90, 55, key="age")
        gender = st.selectbox("Gender", [1, 2], format_func=lambda x: GENDER_MAP[x], key="gender")
        height = st.slider("Height (cm)", 130, 210, 170, key="height")
        weight = st.slider("Weight (kg)", 40, 180, 75, key="weight")
    with c2:
        st.markdown("#### 🩺 Clinical Measurements")
        ap_hi = st.slider("Systolic BP (mm Hg)", 80, 220, 130, key="ap_hi")
        ap_lo = st.slider("Diastolic BP (mm Hg)", 50, 140, 80, key="ap_lo")
        cholesterol = st.selectbox("Cholesterol", [1, 2, 3], format_func=lambda x: CHOL_MAP[x], key="chol")
        gluc = st.selectbox("Glucose", [1, 2, 3], format_func=lambda x: GLUC_MAP[x], key="gluc")
    with c3:
        st.markdown("#### 🚬 Lifestyle")
        smoke = st.selectbox("Smoking", [0, 1], format_func=lambda x: YESNO_MAP[x], key="smoke")
        alco = st.selectbox("Alcohol Intake", [0, 1], format_func=lambda x: YESNO_MAP[x], key="alco")
        active = st.selectbox("Physical Activity", [1, 0], format_func=lambda x: YESNO_MAP[x], key="active")

        bmi_calc = weight / ((height / 100) ** 2)
        st.markdown(f"**Calculated BMI:** `{bmi_calc:.1f}`")
        if bmi_calc < 18.5: st.info("Underweight")
        elif bmi_calc < 25: st.success("Normal weight")
        elif bmi_calc < 30: st.warning("Overweight")
        else: st.error("Obese")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    if st.button("❤️ Predict Heart Disease Risk", type="primary", use_container_width=True):
        input_data = {'age': age, 'gender': gender, 'height': height, 'weight': weight,
                      'ap_hi': ap_hi, 'ap_lo': ap_lo, 'cholesterol': cholesterol,
                      'gluc': gluc, 'smoke': smoke, 'alco': alco, 'active': active}

        sys.path.insert(0, BASE_DIR)
        from src.data_preprocessing import preprocess_single_input

        try:
            X_input = preprocess_single_input(input_data, scaler, metadata['feature_names'])
            prediction = model.predict(X_input)[0]
            probability = model.predict_proba(X_input)[0]
            disease_prob = probability[1]

            c1, c2 = st.columns([1, 1])
            with c1:
                st.plotly_chart(make_gauge(disease_prob), use_container_width=True)
            with c2:
                if disease_prob >= 0.6:
                    risk, css, advice = "🔴 HIGH RISK", "prediction-high", "Strong indicators of cardiovascular disease. Immediate consultation with a cardiologist recommended."
                elif disease_prob >= 0.3:
                    risk, css, advice = "🟡 MODERATE RISK", "prediction-medium", "Some risk factors detected. Lifestyle modifications and regular monitoring recommended."
                else:
                    risk, css, advice = "🟢 LOW RISK", "prediction-low", "Low probability of cardiovascular disease. Continue healthy lifestyle."

                st.markdown(f'<div class="{css}"><h2 style="margin:0;font-size:32px;">{risk}</h2>'
                    f'<p style="font-size:18px;margin-top:12px;">Disease Probability: <b>{disease_prob:.1%}</b></p>'
                    f'<p style="font-size:14px;margin-top:8px;opacity:0.9;">Prediction: <b>{"Cardiovascular Disease" if prediction==1 else "No Disease"}</b></p></div>',
                    unsafe_allow_html=True)
                st.markdown(f'<div class="info-box" style="margin-top:16px;"><b>💡 Recommendation:</b><br>{advice}</div>', unsafe_allow_html=True)

            st.markdown("### 📊 Key Risk Factors")
            factors = []
            if ap_hi >= 160: factors.append(("⚠️ High systolic BP (≥160)", "stage 2 hypertension"))
            elif ap_hi >= 140: factors.append(("⚠️ Elevated systolic BP (≥140)", "stage 1 hypertension"))
            elif ap_hi < 120: factors.append(("✅ Normal systolic BP", "healthy range"))
            if cholesterol == 3: factors.append(("⚠️ Very high cholesterol", "well above normal"))
            elif cholesterol == 1: factors.append(("✅ Normal cholesterol", "healthy range"))
            if bmi_calc >= 30: factors.append(("⚠️ Obese BMI (≥30)", "significant weight risk"))
            elif bmi_calc < 25: factors.append(("✅ Healthy BMI", "normal weight"))
            if age >= 60: factors.append(("⚠️ Age ≥ 60", "advanced age increases risk"))
            if smoke == 1: factors.append(("⚠️ Smoker", "smoking damages blood vessels"))
            if active == 0: factors.append(("⚠️ No physical activity", "sedentary lifestyle"))
            elif active == 1: factors.append(("✅ Physically active", "exercise is protective"))
            if gluc == 3: factors.append(("⚠️ High glucose", "diabetes risk"))
            for f, d in factors:
                st.markdown(f"- **{f}** — _{d}_")

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.exception(e)


# ═══════════════════════════════════════════════════════════
#  PAGE 4: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.markdown('<p class="section-header">📈 Model Performance Dashboard</p>', unsafe_allow_html=True)
    st.markdown("Comparing all 5 trained models on **70,000 patient records**")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    if metadata is None:
        st.error("No trained model found!")
        st.stop()

    st.markdown(f'<div class="metric-card metric-card-green" style="max-width:500px;"><h3>🏆 Best Model</h3>'
        f'<h1>{metadata["best_model_name"]}</h1></div>', unsafe_allow_html=True)

    metrics_df = pd.DataFrame(metadata['metrics'])
    st.markdown("#### 📊 Model Comparison")
    disp = metrics_df.copy()
    for c in ['Accuracy','Precision','Recall','F1-Score','ROC-AUC']:
        if c in disp.columns: disp[c] = disp[c].apply(lambda x: f"{x:.4f}")
    st.dataframe(disp, use_container_width=True, hide_index=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    colors = ['#e53935','#ff6b6b','#f093fb','#4facfe','#43e97b']
    fig = go.Figure()
    for i, m in enumerate(['Accuracy','Precision','Recall','F1-Score','ROC-AUC']):
        if m in metrics_df.columns:
            fig.add_trace(go.Bar(x=metrics_df['Model'], y=metrics_df[m], name=m, marker_color=colors[i]))
    fig.update_layout(barmode='group', height=450, paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter'), yaxis_title='Score', yaxis_range=[0,1.05],
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    st.plotly_chart(fig, use_container_width=True)

    if all_results:
        st.markdown("#### 📈 ROC Curves")
        try:
            from src.data_preprocessing import full_preprocessing_pipeline
            from sklearn.metrics import roc_curve, auc as sk_auc
            fp = os.path.join(BASE_DIR, 'data', 'heart.csv')
            _, X_test, _, y_test, _, _, _ = full_preprocessing_pipeline(fp)

            fig = go.Figure()
            for i, (name, data) in enumerate(all_results.items()):
                if data['y_prob'] is not None:
                    fpr, tpr, _ = roc_curve(y_test, data['y_prob'])
                    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC={sk_auc(fpr,tpr):.3f})",
                        line=dict(color=colors[i%5], width=2)))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name='Random', line=dict(color='gray', dash='dash')))
            fig.update_layout(height=450, paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter'),
                xaxis_title='FPR', yaxis_title='TPR', legend=dict(x=0.55, y=0.05))
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.warning("Could not generate ROC curves.")

        st.markdown("#### 🔢 Confusion Matrices")
        try:
            from sklearn.metrics import confusion_matrix as sk_cm
            cols = st.columns(min(len(all_results), 3))
            for i, (name, data) in enumerate(all_results.items()):
                ci = i % 3
                with cols[ci]:
                    cm = sk_cm(y_test, data['y_pred'])
                    fig = px.imshow(cm, text_auto=True, color_continuous_scale=['#ffffff','#ff6b6b'],
                        title=name, x=['Healthy','Disease'], y=['Healthy','Disease'])
                    fig.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter', size=11),
                        title_font_size=14, coloraxis_showscale=False)
                    st.plotly_chart(fig, use_container_width=True)
                if ci == 2 and i < len(all_results) - 1:
                    cols = st.columns(min(len(all_results) - i - 1, 3))
        except Exception:
            st.warning("Could not generate confusion matrices.")

        st.markdown("#### 🎯 Feature Importance")
        try:
            best_name = metadata['best_model_name']
            bm = all_results[best_name]['model']
            fnames = metadata['feature_names']
            if hasattr(bm, 'feature_importances_'):
                imp = bm.feature_importances_
                sidx = np.argsort(imp)
                fig = go.Figure(go.Bar(x=imp[sidx], y=[fnames[i] for i in sidx], orientation='h',
                    marker=dict(color=imp[sidx], colorscale=[[0,'#ffcdd2'],[1,'#e53935']])))
                fig.update_layout(height=max(400, len(fnames)*22), paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='Inter'), title=f'Feature Importance — {best_name}', xaxis_title='Importance')
                st.plotly_chart(fig, use_container_width=True)
            elif hasattr(bm, 'coef_'):
                coefs = bm.coef_[0]
                sidx = np.argsort(np.abs(coefs))
                clrs = ['#43e97b' if c > 0 else '#ff416c' for c in coefs[sidx]]
                fig = go.Figure(go.Bar(x=coefs[sidx], y=[fnames[i] for i in sidx], orientation='h', marker_color=clrs))
                fig.update_layout(height=max(400, len(fnames)*22), paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='Inter'), title=f'Feature Coefficients — {best_name}', xaxis_title='Coefficient')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Feature importance not available for SVM.")
        except Exception as e:
            st.warning(f"Could not generate feature importance: {e}")
