import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

from src.config import BEST_MODEL_PATH, METRICS_REPORT_PATH

# Page Configuration
st.set_page_config(
    page_title="Workplace Mental Health Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load pipeline model and metrics report
@st.cache_resource
def load_pipeline():
    if not os.path.exists(BEST_MODEL_PATH):
        st.error(f"⚠️ Model artifact missing at `{BEST_MODEL_PATH}`! Please run `python -m src.train` first.")
        return None, None
    pipeline = joblib.load(BEST_MODEL_PATH)
    metrics_report = {}
    if os.path.exists(METRICS_REPORT_PATH):
        with open(METRICS_REPORT_PATH, "r") as f:
            metrics_report = json.load(f)
    return pipeline, metrics_report

pipeline, metrics_report = load_pipeline()

# Title Banner
st.title("🧠 Workplace Mental Health Prediction System")
st.markdown("""
An end-to-end Machine Learning system trained on the **OSMI Mental Health in Tech Survey** to predict an employee's likelihood of seeking mental health treatment (`treatment`).
""")

st.markdown("---")

# Sidebar navigation & model summary
st.sidebar.header("⚙️ Model Configuration")
if metrics_report:
    best_name = metrics_report.get("best_model_name", "Tuned Model")
    st.sidebar.success(f"**Active Model:** {best_name}")
else:
    st.sidebar.info("Model loaded ready for inference.")

# Tab Layout
tab1, tab2 = st.tabs(["🔮 Employee Diagnostic Tool", "📈 Model Comparison & Analytics"])

with tab1:
    st.subheader("📋 Workplace & Demographic Survey Inputs")
    st.write("Provide details about the employee's environment, demographics, and organizational support:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 👤 Demographics & Profile")
        age = st.number_input("Age:", min_value=18, max_value=90, value=30, step=1)
        gender = st.selectbox("Gender:", options=["Male", "Female", "Non-binary", "Other"], index=0)
        country = st.selectbox(
            "Country of Residence:",
            options=["United States", "United Kingdom", "Canada", "Germany", "Ireland", "Netherlands", "Australia", "Other"],
            index=0
        )
        if country == "United States":
            state = st.selectbox(
                "US State:",
                options=["CA", "WA", "NY", "TX", "IL", "MA", "OH", "MI", "PA", "OR", "FL", "Unknown"],
                index=0
            )
        else:
            state = "Not Applicable"
            
        self_employed = st.radio("Self-Employed?", options=["No", "Yes"], index=0)
        family_history = st.radio("Family History of Mental Illness?", options=["No", "Yes"], index=1)
        work_interfere = st.selectbox(
            "If you have a mental health condition, does it interfere with work?",
            options=["Never", "Rarely", "Sometimes", "Often", "Don't know"],
            index=2
        )

    with col2:
        st.markdown("### 🏢 Employer & Support Benefits")
        no_employees = st.selectbox(
            "Company Size (Employees):",
            options=["1-5", "6-25", "26-100", "100-500", "501-1000", "More than 1000"],
            index=2
        )
        tech_company = st.radio("Is your employer primarily a Tech Company?", options=["No", "Yes"], index=1)
        remote_work = st.radio("Do you work Remotely (at least 50%)?", options=["No", "Yes"], index=0)
        benefits = st.selectbox(
            "Does employer provide mental health benefits?",
            options=["Yes", "No", "Don't know", "Not eligible for coverage / N/A"],
            index=0
        )
        care_options = st.selectbox("Do you know the options for care available?", options=["Yes", "No", "Not sure"], index=0)
        wellness_program = st.selectbox("Has employer offered wellness program?", options=["No", "Yes", "Don't know"], index=0)
        seek_help = st.selectbox("Does employer offer resources/help?", options=["No", "Yes", "Don't know"], index=0)

    with col3:
        st.markdown("### 🛡️ Anonymity & Workplace Environment")
        anonymity = st.selectbox("Is anonymity protected if using resources?", options=["Yes", "No", "I don't know"], index=0)
        leave = st.selectbox(
            "Ease of asking for mental health medical leave:",
            options=["Very easy", "Somewhat easy", "Neither easy nor difficult", "Somewhat difficult", "Very difficult", "Don't know"],
            index=1
        )
        mental_health_consequence = st.selectbox("Negative consequences for mental health discussion?", options=["No", "Maybe", "Yes"], index=0)
        phys_health_consequence = st.selectbox("Negative consequences for physical health discussion?", options=["No", "Maybe", "Yes"], index=0)
        coworkers = st.selectbox("Willing to discuss with coworkers?", options=["Some of them", "Yes", "No"], index=0)
        supervisor = st.selectbox("Willing to discuss with direct supervisor?", options=["Yes", "Some of them", "No"], index=0)
        mental_health_interview = st.selectbox("Bring up mental health in job interview?", options=["No", "Maybe", "Yes"], index=0)
        phys_health_interview = st.selectbox("Bring up physical health in job interview?", options=["Maybe", "No", "Yes"], index=0)
        mental_vs_physical = st.selectbox("Employer takes mental health as seriously as physical?", options=["Yes", "Don't know", "No"], index=0)
        obs_consequence = st.selectbox("Have observed negative consequences for peers?", options=["No", "Yes"], index=0)

    st.markdown("---")
    
    if st.button("🔮 Evaluate Mental Health Treatment Likelihood", type="primary"):
        if pipeline is None:
            st.error("Pipeline model not found!")
        else:
            # Construct raw input DataFrame exactly matching dataset columns
            raw_input = pd.DataFrame([{
                "Age": age,
                "Gender": gender,
                "Country": country,
                "state": state,
                "self_employed": self_employed,
                "family_history": family_history,
                "work_interfere": work_interfere,
                "no_employees": no_employees,
                "remote_work": remote_work,
                "tech_company": tech_company,
                "benefits": benefits,
                "care_options": care_options,
                "wellness_program": wellness_program,
                "seek_help": seek_help,
                "anonymity": anonymity,
                "leave": leave,
                "mental_health_consequence": mental_health_consequence,
                "phys_health_consequence": phys_health_consequence,
                "coworkers": coworkers,
                "supervisor": supervisor,
                "mental_health_interview": mental_health_interview,
                "phys_health_interview": phys_health_interview,
                "mental_vs_physical": mental_vs_physical,
                "obs_consequence": obs_consequence
            }])
            
            # Predict using exported pipeline
            # Note: pipeline steps include preprocessor -> classifier
            prediction = pipeline.predict(raw_input)[0]
            
            if hasattr(pipeline, "predict_proba"):
                probs = pipeline.predict_proba(raw_input)[0]
                prob_no, prob_yes = probs[0], probs[1]
            else:
                prob_yes = 1.0 if prediction == 1 else 0.0
                prob_no = 1.0 - prob_yes
                
            st.subheader("🎯 Model Assessment Result")
            
            res_col1, res_col2 = st.columns([1, 1])
            
            with res_col1:
                if prediction == 1:
                    st.error("### 🔴 Stance: HIGH LIKELIHOOD to Seek Treatment")
                    st.write("The model indicates that given family history, work interference, and workplace environment attributes, the employee is **highly likely** to seek mental health treatment.")
                else:
                    st.success("### 🟢 Stance: UNLIKELY to Seek Treatment")
                    st.write("The model indicates that given current workplace attributes and profile, the employee is **unlikely** to seek mental health treatment.")
                    
            with res_col2:
                st.metric(label="Probability of Seeking Treatment (Yes)", value=f"{prob_yes * 100:.2f}%")
                st.metric(label="Probability of Not Seeking Treatment (No)", value=f"{prob_no * 100:.2f}%")
                st.progress(float(prob_yes))

with tab2:
    st.subheader("📈 Benchmark Metrics & Optimization Report")
    
    if metrics_report:
        all_metrics = metrics_report.get("all_model_metrics", {})
        
        # Display comparison table
        table_data = []
        for model_name, m in all_metrics.items():
            table_data.append({
                "Model": model_name,
                "Accuracy (%)": f"{m['Accuracy']*100:.2f}%",
                "F1 Score": f"{m['F1 Score']:.4f}",
                "ROC AUC": f"{m['ROC AUC']:.4f}",
                "Balanced Acc": f"{m['Balanced Accuracy']:.4f}",
                "MCC": f"{m['MCC']:.4f}",
                "Cohen Kappa": f"{m['Cohen Kappa']:.4f}"
            })
            
        df_table = pd.DataFrame(table_data)
        st.dataframe(df_table, use_container_width=True)
        
        st.markdown("### 🔑 Key Predictive Drivers (SHAP & Mutual Information)")
        st.info("""
        1. **`work_interfere`**: Primary driver. Employees experiencing work interference are significantly more likely to seek treatment.
        2. **`family_history`**: Strong genetic/social predictor for seeking care.
        3. **`care_options` & `benefits`**: Clear employer mental health benefits directly increase treatment-seeking behavior.
        4. **`company_support_score`**: Aggregate score of wellness program, anonymity, and seek_help resources.
        """)
    else:
        st.info("Metrics report pending execution.")