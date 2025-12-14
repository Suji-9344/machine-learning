import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("Diabetes Prediction")

age = st.number_input("Age")
bmi = st.number_input("BMI")
glucose = st.number_input("Glucose Level")

if st.button("Predict"):
    result = model.predict([[age, bmi, glucose]])
    st.success("Prediction: " + str(result[0]))
