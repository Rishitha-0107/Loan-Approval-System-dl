
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
# DOWNLOAD MODEL
# =========================================

FILE_ID = "1TnIcnpvL4jEC7Ocw_Cf3NrIjlWB9lDd-"

MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):

    url = f"https://drive.google.com/uc?id={FILE_ID}"

    gdown.download(url, MODEL_PATH, quiet=False)

# =========================================
# LOAD FILES
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
# SIDEBAR
# =========================================

st.sidebar.header("Enter Customer Details")

# =========================================
# NUMERIC INPUTS
# =========================================

LoanID = st.sidebar.number_input("Loan ID", 1000, 9999, 1203)

Age = st.sidebar.slider("Age", 18, 70, 25)

Income = st.sidebar.number_input(
    "Annual Income",
    10000,
    1000000,
    60000
)

LoanAmount = st.sidebar.number_input(
    "Loan Amount",
    1000,
    1000000,
    30000
)

CreditScore = st.sidebar.slider(
    "Credit Score",
    300,
    900,
    650
)

MonthsEmployed = st.sidebar.slider(
    "Months Employed",
    0,
    480,
    36
)

NumCreditLines = st.sidebar.slider(
    "Number of Credit Lines",
    1,
    20,
    5
)

InterestRate = st.sidebar.slider(
    "Interest Rate",
    1.0,
    30.0,
    10.0
)

LoanTerm = st.sidebar.selectbox(
    "Loan Term",
    [12, 24, 36, 48, 60]
)

DTIRatio = st.sidebar.slider(
    "Debt To Income Ratio",
    0.0,
    1.0,
    0.3
)

# =========================================
# CATEGORICAL INPUTS
# =========================================

Education = st.sidebar.selectbox(
    "Education",
    ["High School", "Bachelor", "Master", "PhD"]
)

EmploymentType = st.sidebar.selectbox(
    "Employment Type",
    ["Full-Time", "Part-Time", "Self-Employed", "Unemployed"]
)

MaritalStatus = st.sidebar.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

HasMortgage = st.sidebar.selectbox(
    "Has Mortgage",
    ["No", "Yes"]
)

HasDependents = st.sidebar.selectbox(
    "Has Dependents",
    ["No", "Yes"]
)

LoanPurpose = st.sidebar.selectbox(
    "Loan Purpose",
    ["Home", "Education", "Business", "Car", "Personal"]
)

HasCoSigner = st.sidebar.selectbox(
    "Has Co-Signer",
    ["No", "Yes"]
)

# =========================================
# MANUAL ENCODING
# =========================================

education_map = {
    "High School": 0,
    "Bachelor": 1,
    "Master": 2,
    "PhD": 3
}

employment_map = {
    "Full-Time": 0,
    "Part-Time": 1,
    "Self-Employed": 2,
    "Unemployed": 3
}

marital_map = {
    "Single": 0,
    "Married": 1,
    "Divorced": 2
}

yes_no_map = {
    "No": 0,
    "Yes": 1
}

purpose_map = {
    "Home": 0,
    "Education": 1,
    "Business": 2,
    "Car": 3,
    "Personal": 4
}

# =========================================
# INPUT DATAFRAME
# =========================================

input_data = pd.DataFrame([{
    "LoanID": LoanID,
    "Age": Age,
    "Income": Income,
    "LoanAmount": LoanAmount,
    "CreditScore": CreditScore,
    "MonthsEmployed": MonthsEmployed,
    "NumCreditLines": NumCreditLines,
    "InterestRate": InterestRate,
    "LoanTerm": LoanTerm,
    "DTIRatio": DTIRatio,
    "Education": education_map[Education],
    "EmploymentType": employment_map[EmploymentType],
    "MaritalStatus": marital_map[MaritalStatus],
    "HasMortgage": yes_no_map[HasMortgage],
    "HasDependents": yes_no_map[HasDependents],
    "LoanPurpose": purpose_map[LoanPurpose],
    "HasCoSigner": yes_no_map[HasCoSigner]
}])

# =========================================
# PREDICTION
# =========================================

if st.button("🔍 Predict Loan Status"):

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("❌ High Risk Loan Applicant")

    else:
        st.success("✅ Loan Approved")

# =========================================
# FOOTER
# =========================================

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit & Machine Learning")

