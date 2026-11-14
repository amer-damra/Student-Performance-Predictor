
import streamlit as st
import pickle
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Application title
st.title("🎓 Student Performance Predictor")

st.write(
    "Predict a student's final score using "
    "Multiple Linear Regression."
)

st.divider()

# User inputs
st.subheader("Enter Student Information")

study_hours = st.slider(
    "Daily Study Hours",
    min_value=0.0,
    max_value=12.0,
    value=4.0,
    step=0.5
)

attendance = st.slider(
    "Attendance Percentage",
    min_value=0,
    max_value=100,
    value=80
)

previous_score = st.slider(
    "Previous Exam Score",
    min_value=0,
    max_value=100,
    value=70
)

# Prediction button
if st.button("Predict Final Score", type="primary"):

    input_data = pd.DataFrame([{
        "Study_Hours": study_hours,
        "Attendance": attendance,
        "Previous_Score": previous_score
    }])

    prediction = model.predict(input_data)[0]

    prediction = max(0, min(100, prediction))

    st.success("Prediction Completed!")

    st.metric(
        label="Predicted Final Score",
        value=f"{prediction:.2f} / 100"
    )

    st.info(
        "This prediction is an educational estimate, "
        "not a guaranteed final grade."
    )

st.divider()

st.caption(
    "Powered by Python, Scikit-learn and Streamlit"
)
