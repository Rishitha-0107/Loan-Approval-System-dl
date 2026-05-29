import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# ==============================
# SAFE PATH LOADING
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
features = joblib.load(os.path.join(BASE_DIR, "features.pkl"))
encoders = joblib.load(os.path.join(BASE_DIR, "encoders.pkl"))

# ==============================
# UI CONFIG
# ==============================

st.set_page_config(page_title="Loan Prediction", layout="wide")

st.title("🏦 Loan Approval Prediction System")

st.markdown("### Enter Customer Details")

# ==============================
# INPUT FORM
# ==============================

input_data = {}

for f in features:
    input_data[f] = st.number_input(f, value=0.0)

input_df = pd.DataFrame([input_data])

# ==============================
# PREDICTION
# ==============================

if st.button("Predict Loan Status"):

    prediction = model.predict(input_df)[0]

    st.subheader("Result")

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
