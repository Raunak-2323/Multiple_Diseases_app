import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('parkinsons_model.sav')

# Page title
st.set_page_config(page_title="Parkinson's Disease Predictor")
st.title("🧠 Parkinson's Disease Prediction System")
st.markdown("Enter the required values to check the risk of Parkinson's Disease.")

# Input fields (only 20 features)
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

# Predict
if st.button("Predict Parkinson's Disease"):
    input_data = np.array([[fo, fhi, flo, jitter_percent, rap, ppq, ddp, shimmer, shimmer_db,
                            apq3, apq5, apq, nhr, hnr, rpde, dfa,
                            spread1, spread2, d2, ppe]])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("🔴 The person is likely to have Parkinson's Disease.")
    else:
        st.success("🟢 The person is not likely to have Parkinson's Disease.")
