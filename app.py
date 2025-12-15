import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Student Marks Prediction", page_icon="📘")
st.title("📘 Student Marks Prediction")

MODEL_FILE = "student_marks_model.pkl"

# Check model file (NO uploader)
if not os.path.exists(MODEL_FILE):
    st.error("❌ Model file not found!")
    st.info("Place 'student_marks_model.pkl' in the same folder as app.py")
    st.stop()

# Load model
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

# Input
hours = st.number_input(
    "Enter study hours",
    min_value=0.0,
    value=2.5
)

# Predict
if st.button("Predict Marks"):
    df = pd.DataFrame([[hours]], columns=["Hours"])
    prediction = model.predict(df)
    st.success(f"✅ Predicted Marks: {prediction[0]:.2f}")
