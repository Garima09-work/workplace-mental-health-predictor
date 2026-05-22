import streamlit as st
import pickle
import numpy as np
import os

# Set up the web page title and icon
st.set_page_config(page_title="Workplace Mental Health Predictor", page_icon="🧠", layout="centered")

# App Heading and Introduction
st.title("🧠 Tech Workplace Mental Health Predictor")
st.write("This interactive AI application uses an optimized Random Forest Classifier to predict the likelihood of an employee seeking mental health treatment.")
st.markdown("---")

# Load the saved model and label encoder safely using paths
MODEL_PATH = os.path.join("saved_model", "mental_health_rf_model.pkl")
ENCODER_PATH = os.path.join("saved_model", "label_encoder.pkl")

@st.cache_resource
def load_assets():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        st.error("⚠️ Model or Encoder files missing! Please ensure they are placed inside the 'saved_model' folder.")
        return None, None
    with open(MODEL_PATH, 'rb') as m_file:
        model = pickle.load(m_file)
    with open(ENCODER_PATH, 'rb') as e_file:
        encoder = pickle.load(e_file)
    return model, encoder

model, encoder = load_assets()

if model is not None:
    st.subheader("📋 Employee Workplace Survey Inputs")
    st.write("Please select the attributes for evaluation:")

    # User Interface Dropdowns
    company_size = st.selectbox(
        "1. What is the overall size of the company/organization?",
        options=["1-5", "6-25", "26-100", "101-500", "501-1000", "More than 1000"],
        index=2
    )
    size_mapping = {"1-5": 0, "6-25": 1, "26-100": 2, "101-500": 3, "501-1000": 4, "More than 1000": 5}

    tech_company = st.radio("2. Is your employer primarily a tech company?", options=["No", "Yes"], index=1)

    benefits = st.selectbox(
        "3. Does your employer provide mental health benefits?",
        options=["No", "Yes", "Don't know", "Not eligible for coverage / N/A"]
    )
    benefits_mapping = {"No": 0, "Yes": 1, "Don't know": 2, "Not eligible for coverage / N/A": 3}

    care_options = st.selectbox("4. Do you know the options for mental health care available?", options=["No", "Yes", "Not sure"])
    care_mapping = {"No": 0, "Yes": 1, "Not sure": 2}

    mh_discussion = st.radio("5. Has your employer ever formally discussed mental health?", options=["No", "Yes", "Don't know"])
    discussion_mapping = {"No": 0, "Yes": 1, "Don't know": 2}

    mh_resources = st.selectbox("6. Does your employer offer resources to learn more about mental health?", options=["No", "Yes", "Don't know"])
    resources_mapping = {"No": 0, "Yes": 1, "Don't know": 2}

    anonymity = st.selectbox("7. Is your anonymity protected if you choose to take advantage of resources?", options=["No", "Yes", "I don't know"])
    anonymity_mapping = {"No": 0, "Yes": 1, "I don't know": 2}

    medical_leave = st.selectbox(
        "8. How easy is it to ask for medical leave for a mental health issue?",
        options=["Very easy", "Somewhat easy", "Neither easy nor difficult", "Somewhat difficult", "Very difficult", "Don't know"]
    )
    leave_mapping = {"Very easy": 0, "Somewhat easy": 1, "Neither easy nor difficult": 2, "Somewhat difficult": 3, "Very difficult": 4, "Don't know": 5}

    mh_consequences = st.selectbox("9. Do you think discussing a mental health disorder would have negative consequences?", options=["No", "Yes", "Maybe"])
    consequences_mapping = {"No": 0, "Yes": 1, "Maybe": 2}

    employer_seriousness = st.selectbox("10. Do you feel your employer takes mental health as seriously as physical health?", options=["No", "Yes", "I don't know"])
    seriousness_mapping = {"No": 0, "Yes": 1, "I don't know": 2}

    st.markdown("---")

    if st.button("🔮 Run AI Diagnostics", type="primary"):
        features = [
            size_mapping[company_size],
            1 if tech_company == "Yes" else 0,
            benefits_mapping[benefits],
            care_mapping[care_options],
            discussion_mapping[mh_discussion],
            resources_mapping[mh_resources],
            anonymity_mapping[anonymity],
            leave_mapping[medical_leave],
            consequences_mapping[mh_consequences],
            seriousness_mapping[employer_seriousness]
        ]
        
        input_data = np.array([features])
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        class_labels = {0: "No (Unlikely to seek treatment)", 
                        1: "Maybe (Undecided / Potential trend)", 
                        2: "Yes (Highly likely to seek treatment)"}
        
        st.subheader("🎯 Model Diagnostic Results")
        if prediction == 2:
            st.error(f"**Predicted Stance: {class_labels[prediction]}**")
        elif prediction == 1:
            st.warning(f"**Predicted Stance: {class_labels[prediction]}**")
        else:
            st.success(f"**Predicted Stance: {class_labels[prediction]}**")
            
        st.write("📊 **Model Classification Probabilities:**")
        st.info(f"🔹 Probability of 'No': {probabilities[0]*100:.2f}%")
        st.info(f"🔹 Probability of 'Maybe': {probabilities[1]*100:.2f}%")
        st.info(f"🔹 Probability of 'Yes': {probabilities[2]*100:.2f}%")