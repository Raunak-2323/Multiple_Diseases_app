import streamlit as st
import numpy as np
import joblib

# Load the saved model
model = joblib.load('diabetes_model.sav')

# Title
st.title("🩺 Diabetes Prediction System")
st.write("Enter your health info below to check your diabetes risk:")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=100)
bp = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
skin = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=1, max_value=120, value=30)

# Predict button
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)
    result = "🔴 Diabetic" if prediction[0] == 1 else "🟢 Not Diabetic"
    st.subheader("Prediction Result:")
    st.success(result)
