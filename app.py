import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gdown

# =========================================
# GOOGLE DRIVE FILE IDS
# =========================================

MODEL_ID = "1TnIcnpvL4jEC7Ocw_Cf3NrIjlWB9lDd-"

# =========================================
# FILE PATHS
# =========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "risk_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
PCA_PATH = os.path.join(BASE_DIR, "pca.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_names.pkl")

# =========================================
# DOWNLOAD MODEL IF NOT EXISTS
# =========================================

if not os.path.exists(MODEL_PATH):

    url = f"https://drive.google.com/uc?id={MODEL_ID}"

    gdown.download(url, MODEL_PATH, quiet=False)

# =========================================
# LOAD FILES
# =========================================

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
pca = joblib.load(PCA_PATH)
feature_names = joblib.load(FEATURES_PATH)

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)

# =========================================
# TITLE
# =========================================

st.title("🏦 Loan Approval Prediction System")
st.markdown("### AI-Powered Banking Risk Assessment")

st.divider()

# =========================================
# SIDEBAR INPUTS
# =========================================

st.sidebar.header("Customer Details")

input_data = {}

for feature in feature_names:

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

    # Scaling
    scaled_data = scaler.transform(input_df)

    # PCA
    pca_data = pca.transform(scaled_data)

    # Prediction
    prediction = model.predict(pca_data)[0]

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
