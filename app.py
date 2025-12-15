import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Student Marks Prediction", page_icon="📘")
st.title("📘 Student Marks Prediction")

MODEL_FILE = "student_marks_model.pkl"

# ----------------------------
# Auto-create model if missing
# ----------------------------
if not os.path.exists(MODEL_FILE):
    # Sample training data
    data = {
        "Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "Marks": [35, 40, 50, 60, 65, 70, 75, 85, 90]
    }

    df = pd.DataFrame(data)

    X = df[["Hours"]]
    y = df["Marks"]

    model = LinearRegression()
    model.fit(X, y)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

# ----------------------------
# Load model
# ----------------------------
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

# ----------------------------
# User Input
# ----------------------------
hours = st.number_input(
    "Enter study hours",
    min_value=0.0,
    value=2.5
)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Marks"):
    df = pd.DataFrame([[hours]], columns=["Hours"])
    prediction = model.predict(df)
    st.success(f"✅ Predicted Marks: {prediction[0]:.2f}")
