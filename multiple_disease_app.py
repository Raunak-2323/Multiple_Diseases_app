import streamlit as st
import numpy as np
import joblib

# Load models
diabetes_model = joblib.load('diabetes_model.sav')
heart_model = joblib.load('heart_model.sav')
parkinsons_model = joblib.load('parkinsons_model.sav')

# Page setup
st.set_page_config(page_title="Multiple Disease Prediction System")
st.title("🧬 Multiple Disease Prediction System")

# Sidebar for selection
selected_disease = st.sidebar.selectbox("Choose Disease to Predict", 
                                        ["Diabetes", "Heart Disease", "Parkinson's Disease"])

# Diabetes Prediction
if selected_disease == "Diabetes":
    st.header("🩸 Diabetes Prediction")
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose Level", 0, 200, 100)
    bp = st.number_input("Blood Pressure", 0, 150, 70)
    skin = st.number_input("Skin Thickness", 0, 100, 20)
    insulin = st.number_input("Insulin", 0, 900, 80)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.number_input("Age", 10, 100, 30)

    if st.button("Predict Diabetes"):
        input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        result = diabetes_model.predict(input_data)[0]
        st.success("🟢 Not Diabetic" if result == 0 else "🔴 Diabetic")

# Heart Disease Prediction
elif selected_disease == "Heart Disease":
    st.header("❤️ Heart Disease Prediction")
    age = st.number_input("Age", 10, 100, 45)
    sex = st.number_input("Sex (1=Male, 0=Female)", 0, 1)
    cp = st.number_input("Chest Pain Type (0-3)", 0, 3)
    trestbps = st.number_input("Resting BP", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.number_input("Fasting Blood Sugar > 120 (1=True, 0=False)", 0, 1)
    restecg = st.number_input("Rest ECG (0-2)", 0, 2)
    thalach = st.number_input("Max Heart Rate", 60, 210, 150)
    exang = st.number_input("Exercise Induced Angina (1=Yes, 0=No)", 0, 1)
    oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
    slope = st.number_input("Slope (0-2)", 0, 2)
    ca = st.number_input("CA (0-4)", 0, 4)
    thal = st.number_input("Thal (1=Normal, 2=Fixed Defect, 3=Reversible)", 0, 3)

    if st.button("Predict Heart Disease"):
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
                                oldpeak, slope, ca, thal]])
        result = heart_model.predict(input_data)[0]
        st.success("🟢 No Heart Disease" if result == 0 else "🔴 Heart Disease Detected")

# Parkinson’s Prediction
else:
    st.header("🧠 Parkinson's Disease Prediction")
    fo = st.number_input("MDVP:Fo(Hz)", value=120.0)
    fhi = st.number_input("MDVP:Fhi(Hz)", value=130.0)
    flo = st.number_input("MDVP:Flo(Hz)", value=110.0)
    jitter_percent = st.number_input("Jitter (%)", value=0.005)
    rap = st.number_input("Jitter (RAP)", value=0.003)
    ppq = st.number_input("Jitter (PPQ)", value=0.004)
    ddp = st.number_input("Jitter: DDP", value=0.009)
    shimmer = st.number_input("Shimmer", value=0.03)
    shimmer_db = st.number_input("Shimmer (dB)", value=0.3)
    apq3 = st.number_input("Shimmer: APQ3", value=0.015)
    apq5 = st.number_input("Shimmer: APQ5", value=0.02)
    apq = st.number_input("Shimmer: APQ", value=0.025)
    nhr = st.number_input("NHR", value=0.03)
    hnr = st.number_input("HNR", value=20.0)
    rpde = st.number_input("RPDE", value=0.4)
    dfa = st.number_input("DFA", value=0.7)
    spread1 = st.number_input("Spread1", value=-6.0)
    spread2 = st.number_input("Spread2", value=0.1)
    d2 = st.number_input("D2", value=2.5)
    ppe = st.number_input("PPE", value=0.3)

    if st.button("Predict Parkinson's"):
        input_data = np.array([[fo, fhi, flo, jitter_percent, rap, ppq, ddp, shimmer, shimmer_db,
                                apq3, apq5, apq, nhr, hnr, rpde, dfa,
                                spread1, spread2, d2, ppe]])
        result = parkinsons_model.predict(input_data)[0]
        st.success("🟢 No Parkinson's Detected" if result == 0 else "🔴 Parkinson's Detected")
