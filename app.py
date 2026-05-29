import streamlit as st
import numpy as np
import pandas as pd
import joblib
import gdown
import os

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
# LOAD MODEL
# =========================================

model = joblib.load(MODEL_PATH)

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

st.markdown(
    "### AI-Powered Banking Risk Assessment"
)

st.divider()

# =========================================
# INPUT SECTION
# =========================================

st.sidebar.header("Customer Details")

feature_count = model.n_features_in_

input_values = []

for i in range(feature_count):

    value = st.sidebar.number_input(
        f"Feature {i+1}",
        value=0.0
    )

    input_values.append(value)

# =========================================
# PREDICTION
# =========================================

if st.button("🔍 Predict Loan Status"):

    input_array = np.array([input_values])

    prediction = model.predict(input_array)[0]

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
