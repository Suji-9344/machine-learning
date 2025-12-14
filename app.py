import streamlit as st
import pickle
import numpy as np

# Load trained model
try:
    with open("diabetes_model.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'diabetes_model.pkl' not found. Please check the path.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("Diabetes Prediction App")

st.markdown("### Enter Patient Details:")

# User inputs
pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, step=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, step=1)
blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, step=1)
skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, step=1)
insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=1000, step=1)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, step=0.1)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.01)
age = st.number_input("Age", min_value=0, max_value=120, step=1)

# Prediction
if st.button("Predict Diabetes"):
    try:
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin,
                                bmi, diabetes_pedigree, age]])
        prediction = model.predict(input_data)[0]
        if prediction == 1:
            st.error("The patient is likely to have diabetes.")
        else:
            st.success("The patient is not likely to have diabetes.")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
