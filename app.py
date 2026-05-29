
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gdown
import os

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)

# =========================================
# DOWNLOAD MODEL FROM GOOGLE DRIVE
# =========================================

FILE_ID = "1TnIcnpvL4jEC7Ocw_Cf3NrIjlWB9lDd-"

MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):

    url = f"https://drive.google.com/uc?id={FILE_ID}"

    gdown.download(
        url,
        MODEL_PATH,
        quiet=False
    )

# =========================================
# LOAD MODEL + FEATURES
# =========================================

model = joblib.load(MODEL_PATH)

features = joblib.load("features.pkl")

# =========================================
# TITLE
# =========================================

st.title("🏦 Loan Approval Prediction System")

st.markdown(
    "### AI-Powered Banking Risk Assessment Dashboard"
)

st.divider()

# =========================================
# SIDEBAR INPUTS
# =========================================

st.sidebar.header("Enter Customer Details")

input_data = {}

for feature in features:

    input_data[feature] = st.sidebar.number_input(
        feature,
        value=0.0
    )

# =========================================
# DATAFRAME
# =========================================

input_df = pd.DataFrame([input_data])

# =========================================
# PREDICTION
# =========================================

if st.button("🔍 Predict Loan Status"):

    prediction = model.predict(input_df)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("✅ Loan Approved")

    else:
        st.error("❌ Loan Rejected")

# =========================================
# FOOTER
# =========================================

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit & Machine Learning")

