import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

from src.config import (
    RF_MODEL_PATH, XGB_MODEL_PATH, ENCODER_PATH,
    SIZE_MAPPING, TECH_MAPPING, BENEFITS_MAPPING, CARE_MAPPING,
    DISCUSSION_MAPPING, RESOURCES_MAPPING, ANONYMITY_MAPPING,
    LEAVE_MAPPING, CONSEQUENCES_MAPPING, SERIOUSNESS_MAPPING, CLASS_LABELS
)

# Page configuration
st.set_page_config(page_title="Workplace Mental Health Predictor", page_icon="🧠", layout="centered")

st.title("🧠 Tech Workplace Mental Health Predictor")
st.write("Predict employee likelihood of seeking mental health treatment based on workplace environment attributes.")
st.markdown("---")

# Sidebar - Model Selection
st.sidebar.header("⚙️ Model Configuration")
selected_model_type = st.sidebar.selectbox(
    "Choose Machine Learning Model:",
    options=["Random Forest Classifier", "XGBoost Classifier"]
)

# Load selected model and encoder safely
@st.cache_resource
def load_model(model_type):
    model_path = XGB_MODEL_PATH if model_type == "XGBoost Classifier" else RF_MODEL_PATH
    if not os.path.exists(model_path) or not os.path.exists(ENCODER_PATH):
        st.error(f"⚠️ Model artifact missing at `{model_path}`! Please run `python -m src.train` first.")
        return None, None
    with open(model_path, 'rb') as m_file:
        model = pickle.load(m_file)
    with open(ENCODER_PATH, 'rb') as e_file:
        encoder = pickle.load(e_file)
    return model, encoder

model, encoder = load_model(selected_model_type)

st.sidebar.info(f"**Active Model:** {selected_model_type}")

if model is not None:
    st.subheader("📋 Employee Workplace Survey Inputs")
    st.write("Please select the attributes for evaluation:")

    # Inputs
    company_size = st.selectbox(
        "1. What is the overall size of the company/organization?",
        options=["1-5", "6-25", "26-100", "101-500", "501-1000", "More than 1000"],
        index=2
    )

    tech_company = st.radio("2. Is your employer primarily a tech company?", options=["No", "Yes"], index=1)

    benefits = st.selectbox(
        "3. Does your employer provide mental health benefits?",
        options=["No", "Yes", "Don't know", "Not eligible for coverage / N/A"]
    )

    care_options = st.selectbox("4. Do you know the options for mental health care available?", options=["No", "Yes", "Not sure"])

    mh_discussion = st.radio("5. Has your employer ever formally discussed mental health?", options=["No", "Yes", "Don't know"])

    mh_resources = st.selectbox("6. Does your employer offer resources to learn more about mental health?", options=["No", "Yes", "Don't know"])

    anonymity = st.selectbox("7. Is your anonymity protected if you choose to take advantage of resources?", options=["No", "Yes", "I don't know"])

    medical_leave = st.selectbox(
        "8. How easy is it to ask for medical leave for a mental health issue?",
        options=["Very easy", "Somewhat easy", "Neither easy nor difficult", "Somewhat difficult", "Very difficult", "Don't know"]
    )

    mh_consequences = st.selectbox("9. Do you think discussing a mental health disorder would have negative consequences?", options=["No", "Yes", "Maybe"])

    employer_seriousness = st.selectbox("10. Do you feel your employer takes mental health as seriously as physical health?", options=["No", "Yes", "I don't know"])

    st.markdown("---")

    if st.button("🔮 Run AI Diagnostics", type="primary"):
        features = [
            SIZE_MAPPING[company_size],
            TECH_MAPPING[tech_company],
            BENEFITS_MAPPING[benefits],
            CARE_MAPPING[care_options],
            DISCUSSION_MAPPING[mh_discussion],
            RESOURCES_MAPPING[mh_resources],
            ANONYMITY_MAPPING[anonymity],
            LEAVE_MAPPING[medical_leave],
            CONSEQUENCES_MAPPING[mh_consequences],
            SERIOUSNESS_MAPPING[employer_seriousness]
        ]

        feature_names = [
            'no_employees', 'tech_company', 'benefits', 'care_options',
            'seek_help', 'wellness_program', 'anonymity', 'leave',
            'mental_health_consequence', 'mental_vs_physical'
        ]

        input_df = pd.DataFrame([features], columns=feature_names)
        prediction = int(model.predict(input_df)[0])

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
        else:
            probabilities = [0.0, 0.0, 0.0]

        st.subheader(f"🎯 Model Diagnostic Results ({selected_model_type})")
        if prediction == 2:
            st.error(f"**Predicted Stance: {CLASS_LABELS[prediction]}**")
        elif prediction == 1:
            st.warning(f"**Predicted Stance: {CLASS_LABELS[prediction]}**")
        else:
            st.success(f"**Predicted Stance: {CLASS_LABELS[prediction]}**")

        st.write("📊 **Model Classification Probabilities:**")
        st.info(f"🔹 Probability of 'No': {probabilities[0]*100:.2f}%")
        st.info(f"🔹 Probability of 'Maybe': {probabilities[1]*100:.2f}%")
        st.info(f"🔹 Probability of 'Yes': {probabilities[2]*100:.2f}%")