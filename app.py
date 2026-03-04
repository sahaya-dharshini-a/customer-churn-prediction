import streamlit as st
from src.predict import predict

st.title("Customer Churn Prediction System")
st.markdown("Machine Learning based churn prediction using RandomForest")
st.write("Enter customer details below:")

tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 500.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

input_data = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Contract": contract
}

if st.button("Predict"):
    prediction, probability = predict(input_data)

    if prediction == 1:
        st.error(f"Customer is likely to churn (Probability: {probability:.2f})")
    else:
        st.success(f"Customer is likely to stay (Probability: {probability:.2f})")