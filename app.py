import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os

st.title("💼 Salary Prediction App")

DATA_FILE = "salary_data.pkl"

# Load dataset
if not os.path.exists(DATA_FILE):
    st.error("❌ salary_data.pkl file not found")
    st.stop()

with open(DATA_FILE, "rb") as f:
    df = pickle.load(f)

st.success("✅ Dataset loaded")

# Show column names (VERY IMPORTANT)
st.write("📌 Dataset Columns:", list(df.columns))
st.write(df.head())

# Auto-detect columns
# First numeric column -> feature
# Last numeric column -> target
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) < 2:
    st.error("❌ Dataset must contain at least 2 numeric columns")
    st.stop()

X_col = numeric_cols[0]
y_col = numeric_cols[-1]

st.info(f"Using Feature: {X_col}")
st.info(f"Using Target: {y_col}")

X = df[[X_col]]
y = df[y_col]

# Train model
model = LinearRegression()
model.fit(X, y)

# User input
experience = st.number_input(
    f"Enter {X_col}",
    min_value=float(df[X_col].min()),
    max_value=float(df[X_col].max()),
    value=float(df[X_col].mean())
)

# Prediction
if st.button("Predict Salary"):
    prediction = model.predict([[experience]])
    st.success(f"💰 Predicted {y_col}: ₹ {prediction[0]:,.2f}")
