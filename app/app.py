
import streamlit as st
import joblib
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "fraud_detection_model.pkl"
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("💳 Financial Fraud Detection System")

st.write(
    "Machine Learning based system for detecting potentially "
    "fraudulent credit card transactions."
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.header("📊 About the Model")

    st.write("""
    **Algorithm:** Random Forest

    **Number of Trees:** 200

    **Maximum Depth:** 20

    **Minimum Samples per Leaf:** 2

    **Class Weight:** Balanced
    """)

    st.divider()

    st.info(
        "Enter the transaction features and click "
        "'Check Transaction' to get a prediction."
    )


# --------------------------------------------------
# TRANSACTION INFORMATION
# --------------------------------------------------

st.subheader("💰 Transaction Information")

col1, col2 = st.columns(2)

with col1:
    Time = st.number_input(
        "Transaction Time",
        min_value=0.0,
        value=0.0
    )

with col2:
    Amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=0.0
    )


# --------------------------------------------------
# V1 - V28 FEATURES
# --------------------------------------------------

st.subheader("🔢 Transaction Features")

features = {}

columns = st.columns(4)

for i in range(1, 29):

    with columns[(i - 1) % 4]:

        features[f"V{i}"] = st.number_input(
            f"V{i}",
            value=0.0,
            format="%.6f"
        )


st.divider()


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button(
    "🔍 Check Transaction",
    use_container_width=True
):

    # Create input dictionary
    input_data = {
        "Time": Time,
        **features,
        "Amount": Amount
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Get fraud probability
    fraud_probability = model.predict_proba(input_df)[0][1]

    fraud_percentage = fraud_probability * 100


    # --------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------

    st.subheader("📋 Prediction Result")

    if prediction == 1:

        st.error("🚨 FRAUDULENT TRANSACTION")

        st.metric(
            "Fraud Probability",
            f"{fraud_percentage:.2f}%"
        )

    else:

        st.success("✅ LEGITIMATE TRANSACTION")

        st.metric(
            "Fraud Probability",
            f"{fraud_percentage:.2f}%"
        )


    # --------------------------------------------------
    # RISK LEVEL
    # --------------------------------------------------

    if fraud_probability >= 0.70:

        risk_level = "🔴 HIGH RISK"

    elif fraud_probability >= 0.30:

        risk_level = "🟠 MEDIUM RISK"

    else:

        risk_level = "🟢 LOW RISK"


    st.write(f"### Risk Level: {risk_level}")


    # --------------------------------------------------
    # PROBABILITY BAR
    # --------------------------------------------------

    st.write("Fraud Risk")

    st.progress(
        min(int(fraud_percentage), 100)
    )


    # --------------------------------------------------
    # TRANSACTION SUMMARY
    # --------------------------------------------------

    st.subheader("📌 Transaction Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:
        st.metric(
            "Transaction Amount",
            f"{Amount:.2f}"
        )

    with summary_col2:
        st.metric(
            "Fraud Probability",
            f"{fraud_percentage:.2f}%"
        )

    with summary_col3:

        if prediction == 1:
            st.metric("Prediction", "FRAUD")
        else:
            st.metric("Prediction", "LEGITIMATE")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Financial Fraud Detection System | "
    "Machine Learning Project"
)

