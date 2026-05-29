import streamlit as st
import numpy as np
import joblib
import pandas as pd
import os

# =====================================================
# LOAD MODEL & SUPPORT FILES
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "random_forest.pkl"))
features = joblib.load(os.path.join(BASE_DIR, "features.pkl"))
label_encoders = joblib.load(os.path.join(BASE_DIR, "label_encoders.pkl"))

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="🏢",
    layout="wide"
)

# =====================================================
# HEADER
# =====================================================

st.title("🏢 Employee Attrition Prediction System")
st.markdown("### AI-Based HR Analytics Dashboard")

st.divider()

# =====================================================
# SIDEBAR INPUT
# =====================================================

st.sidebar.header("Enter Employee Details")

input_data = {}

for feature in features:
    input_data[feature] = st.sidebar.number_input(
        feature,
        value=0.0
    )

input_df = pd.DataFrame([input_data])

# =====================================================
# PREDICTION BUTTON
# =====================================================

if st.button("🔍 Predict Attrition Risk"):

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("📊 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error("⚠️ Employee is Likely to Leave")
        else:
            st.success("✅ Employee is Likely to Stay")

    with col2:
        st.info(f"🔥 Attrition Probability: {probability * 100:.2f}%")

    # Risk Meter
    st.progress(float(probability))

    # Extra Insight
    st.markdown("### 📌 HR Insight")
    if probability > 0.7:
        st.warning("High Risk Employee - Immediate Retention Action Needed")
    elif probability > 0.4:
        st.info("Medium Risk Employee - Monitor Closely")
    else:
        st.success("Low Risk Employee - Stable")

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")
st.markdown("💡 Built using Machine Learning + Streamlit")
