import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Student Marks Prediction", page_icon="📘")
st.title("📘 Student Marks Prediction")

MODEL_FILE = "student_marks_model.pkl"

# Check if model exists
if not os.path.exists(MODEL_FILE):
    st.error("Model file not found!")
    st.info("Please upload student_marks_model.pkl in this app folder.")

    uploaded_file = st.file_uploader("Upload model file", type=["pkl"])
    if uploaded_file is not None:
        with open(MODEL_FILE, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("Model uploaded successfully. Please refresh the app.")
    st.stop()

# Load model safely
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

# Input
hours = st.number_input("Enter study hours", min_value=0.0, value=2.5)

# Predict
if st.button("Predict Marks"):
    df = pd.DataFrame([[hours]], columns=["Hours"])
    prediction = model.predict(df)
    st.success(f"Predicted Marks: {prediction[0]:.2f}")
