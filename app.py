import streamlit as st
import pickle
import numpy as np

# Load the trained model
try:
    with open("salary_data.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'salary_data.pkl' not found. Please check the path.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# App title
st.title("Salary Prediction App")

# Input from user
years_of_experience = st.number_input(
    "Enter Years of Experience",
    min_value=0.0,
    max_value=50.0,
    step=0.1
)

# Prediction button
if st.button("Predict Salary"):
    try:
        # Ensure input is valid
        if years_of_experience < 0:
            st.warning("Years of experience cannot be negative.")
        else:
            input_data = np.array([[years_of_experience]])
            predicted_salary = model.predict(input_data)[0]
            st.success(f"Predicted Salary: ₹{predicted_salary:,.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

