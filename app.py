import streamlit as st
import pickle
import numpy as np

# Load model
with open("logistic_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Logistic Regression Prediction App")

# Input fields
feature1 = st.number_input("Enter Feature 1")
feature2 = st.number_input("Enter Feature 2")

if st.button("Predict"):
    input_data = np.array([[feature1, feature2]])
    prediction = model.predict(input_data)
    st.success(f"Prediction: {prediction[0]}")
