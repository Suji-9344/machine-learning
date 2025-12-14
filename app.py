import streamlit as st
import pickle
import numpy as np
import os

st.title("🩺 Diabetes Prediction App")

# Show files (for debugging – you can remove later)
st.write("Available files:", os.listdir())

# Correct model file name
model_path = "logistic_regression_model.pkl"

# Load model safely
if not os.path.exists(model_path):
    st.error("❌ Model file not found: logistic_regression_model.pkl")
else:
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    st.success("✅ Model loaded successfully")

    # Inputs (ORDER MUST MATCH TRAINING)
    age = st.number_input("Age", min_value=1, max_value=120, value=25)
    mass = st.number_input("Body Mass Index (BMI)", min_value=0.0, value=25.0)
    insu = st.number_input("Insulin Level", min_value=0.0, value=80.0)
    plas = st.number_input("Plasma Glucose", min_value=0.0, value=120.0)

    if st.button("Predict"):
        input_data = np.array([[age, mass, insu, plas]])
        prediction = model.predict(input_data)

        if prediction[0] == "tested_positive":
            st.error("⚠️ Patient is likely Diabetic")
        else:
            st.success("✅ Patient is likely NOT Diabetic")
