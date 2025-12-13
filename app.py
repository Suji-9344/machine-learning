
import pickle
import numpy as np

@st.cache_resource
def load_model():
    with open("logistic_regression_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("Logistic Regression Prediction App")

f1 = st.number_input("Feature 1")
f2 = st.number_input("Feature 2")
f3 = st.number_input("Feature 3")
f4 = st.number_input("Feature 4")

if st.button("Predict"):
    input_data = np.array([[f1, f2, f3, f4]])
    prediction = model.predict(input_data)
    st.success(f"Prediction: {prediction[0]}")
