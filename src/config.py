import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "survey.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_mental_health_pipeline.pkl")
METRICS_REPORT_PATH = os.path.join(MODELS_DIR, "model_metrics_report.json")

# Target column
TARGET_COL = "treatment"

# Columns to remove (no predictive value / unstandardized text)
DROP_COLS = ["Timestamp", "comments"]

# Feature Categorizations
NUMERIC_COLS = ["Age", "company_support_score", "consequence_concern_score", "workplace_comfort_score"]

ORDINAL_COLS = ["no_employees", "leave", "work_interfere"]

# Define explicit ordering for Ordinal Encoded features
ORDINAL_CATEGORIES = [
    ["1-5", "6-25", "26-100", "100-500", "501-1000", "More than 1000"], # no_employees
    ["Very easy", "Somewhat easy", "Don't know", "Neither easy nor difficult", "Somewhat difficult", "Very difficult"], # leave
    ["Never", "Rarely", "Sometimes", "Often", "Don't know"] # work_interfere
]

NOMINAL_COLS = [
    "Gender", "Country", "state", "self_employed", "family_history",
    "remote_work", "tech_company", "benefits", "care_options",
    "wellness_program", "seek_help", "anonymity",
    "mental_health_consequence", "phys_health_consequence",
    "coworkers", "supervisor", "mental_health_interview",
    "phys_health_interview", "mental_vs_physical", "obs_consequence"
]
