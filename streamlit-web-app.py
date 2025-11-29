################################################################################
# Deploying ML Models via Streamlit Web App Development
################################################################################

# Import libraries
import streamlit as st
import pandas as pd
import joblib

# Load model pipeline object
model = joblib.load("model.joblib")

# Add title and instructions
st.title("Purchase Prediction Model")
st.subheader("Enter customer info and submit for likelihood to purchase")

# Age input
age = st.number_input(label = "01. Enter the customer's age",
                      min_value = 18,
                      max_value = 120,
                      value = 35)

# Gender input
gender = st.radio(label = "02. Enter the customer's gender",
                  options = ["M", "F"])

# Credit score input
credit_score = st.number_input(label = "03. Enter the customer's credit score",
                      min_value = 0,
                      max_value = 1000,
                      value = 500)

# Submit inputs to model
if st.button("Submit Inputs for Prediction"):
    # Store data in DF for prediction
    new_data = pd.DataFrame({"age" : [age], "gender" : [gender],
                             "credit_score" : [credit_score]})
    
    # Apply model pipeline to input data and extract prob prediction
    pred_proba = model.predict_proba(new_data)[0][1]
    
    # Output prediction
    st.subheader(f"Based on these customer attributes, our model predicts a purchase probability of {pred_proba : .0%}")

