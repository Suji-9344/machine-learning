import streamlit as st
import pickle
import numpy as np

# Load the trained logistic regression model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Logistic Regression Prediction App")
st.write("Predict outcome based on a single input feature.")

# Input from user
user_input = st.number_input("Enter the value for input feature:", value=0.0)

# Make prediction button
if st.button("Predict"):
    # Reshape input to 2D array for the model
    input_array = np.array(user_input).reshape(1, -1)
    
    # Get prediction
    prediction = model.predict(input_array)[0]
    prediction_prob = model.predict_proba(input_array)[0][1]

    # Display result
    st.write(f"Predicted Class: {prediction}")
    st.write(f"Probability of class 1: {prediction_prob:.2f}")

