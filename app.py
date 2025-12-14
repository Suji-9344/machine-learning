import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os

# App title
st.title("💼 Salary Prediction App")

DATA_FILE = "salary_data.pkl"

# Check dataset file
if not os.path.exists(DATA_FILE):
    st.error("❌ salary_data.pkl file not found")
    st.stop()

# Load dataset (NOT model)
with open(DATA_FILE, "rb") as f:
    df = pickle.load(f)

st.success("✅ Dataset loaded successfully")

# Show dataset preview (optional)
st.write("Dataset Preview:", df.head())

# Feature & target (adjust if column names differ)
X = df[['YearsExperience']]
y = df['Salary']

# Train model
model = LinearRegression()
model.fit(X, y)

st.success("✅ Model trained successfully")

# User input
experience = st.number_input(
    "Years of Experience",
    min_value=0.0,
    max_value=50.0,
    value=1.0,
    step=0.1
)

# Prediction
if st.button("Predict Salary"):
    prediction = model.predict(np.array([[experience]]))
    st.success(f"💰 Predicted Salary: ₹ {prediction[0]:,.2f}")
