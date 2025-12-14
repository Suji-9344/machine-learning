import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🩺 Diabetes Prediction App")
st.write("Enter patient details to predict diabetes")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=25)
mass = st.number_input("Body Mass Index (BMI)", min_value=0.0, value=25.0)
insu = st.number_input("Insulin Level", min_value=0.0, value=80.0)
plas = st.number_input("Plasma Glucose", min_value=0.0, value=120.0)

# Predict button
if st.button("Predict"):
    input_data = np.array([[age, mass, insu, plas]])
    prediction = model.predict(input_data)

    if prediction[0] == "tested_positive":
        st.error("⚠️ The patient is likely to have Diabetes")
    else:
        st.success("✅ The patient is not likely to have Diabetes")
