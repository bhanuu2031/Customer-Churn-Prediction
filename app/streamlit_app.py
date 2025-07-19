import sys
import os
sys.path.append(os.path.abspath("../src")) 

import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from preprocessing import load_and_preprocess 

model = joblib.load("models/churn_model.pkl")
metrics = joblib.load("models/metrics.pkl")

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉")
st.title("📉 Customer Churn Prediction")
st.markdown("Enter customer details below to predict **churn risk**:")

st.sidebar.title("📊 Model Metrics")
st.sidebar.metric("Accuracy", f"{metrics['accuracy']:.2%}")
st.sidebar.metric("AUC", f"{metrics['roc_auc']:.2%}")
st.sidebar.metric("F1 Score", f"{metrics['f1_score']:.2%}")

gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Has Partner?", ["Yes", "No"])
dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)
tenure = st.slider("Tenure (months)", 0, 72, 24)
monthly_charges = st.slider("Monthly Charges", 0.0, 150.0, 70.0)
total_charges = st.slider("Total Charges", 0.0, 10000.0, 2500.0)

input_dict = {
    "gender": [gender],
    "SeniorCitizen": [1 if senior == "Yes" else 0],
    "Partner": [partner],
    "Dependents": [dependents],
    "PhoneService": [phone_service],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
    "Contract": [contract],
    "PaperlessBilling": [paperless_billing],
    "PaymentMethod": [payment_method],
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges]
}

input_df = pd.DataFrame(input_dict)

df_encoded = load_and_preprocess(None, custom_df=input_df)

if st.button("Predict Churn"):
    pred = model.predict(df_encoded)[0]
    proba = model.predict_proba(df_encoded)[0][1]

    st.subheader("🔮 Prediction Result:")
    if pred == 1:
        st.error(f"🚨 Likely to Churn (Probability: {proba:.2f})")
    else:
        st.success(f"✅ Likely to Stay (Probability of Churn: {proba:.2f})")

    with st.expander("📊 See Model Input (Preprocessed)"):
        st.dataframe(df_encoded)

    st.subheader("🔍 SHAP Feature Explanation")
    explainer = shap.Explainer(model)
    shap_values = explainer(df_encoded)

    fig, ax = plt.subplots(figsize=(10, 3))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)
