import streamlit as st
import numpy as np
import joblib
import os
import gdown

# =========================================
# DOWNLOAD MODEL FROM GOOGLE DRIVE
# =========================================

FILE_ID = "1TnIcnpvL4jEC7Ocw_Cf3NrIjlWB9lDd-"
MODEL_PATH = "model.pkl"

# Download model if not already downloaded
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
    "### AI-Powered Banking Risk Assessment Dashboard"
)

st.divider()

# =========================================
# SIDEBAR INPUTS
# =========================================

st.sidebar.header("Enter Customer Details")

Age = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=25
)

Income = st.sidebar.number_input(
    "Income",
    min_value=0,
    max_value=1000000,
    value=50000
)

LoanAmount = st.sidebar.number_input(
    "Loan Amount",
    min_value=0,
    max_value=1000000,
    value=100000
)

CreditScore = st.sidebar.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=650
)

Experience = st.sidebar.number_input(
    "Years of Experience",
    min_value=0,
    max_value=50,
    value=2
)

# =========================================
# CREATE INPUT ARRAY
# =========================================

input_data = np.array([[
    Age,
    Income,
    LoanAmount,
    CreditScore,
    Experience
]])

# =========================================
# PREDICTION
# =========================================

if st.button("🔍 Predict Loan Status"):

    prediction = model.predict(input_data)[0]

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
