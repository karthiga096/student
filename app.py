
import streamlit as st
import pandas as pd
import pickle

# Page setup
st.set_page_config(page_title="Student Marks Prediction", page_icon="📘")

st.title("📘 Student Marks Prediction")
st.write("Predict student marks based on study hours")

# Load model
with open("student_marks_model.pkl", "rb") as f:
    model = pickle.load(f)

# User input
hours = st.number_input("Enter study hours", min_value=0.0, value=2.5)

# Prediction
if st.button("Predict"):
    input_df = pd.DataFrame([[hours]], columns=["Hours"])
    result = model.predict(input_df)
    st.success(f"Predicted Marks: {result[0]:.2f}")
