import os

# File paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "survey.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

RF_MODEL_PATH = os.path.join(MODELS_DIR, "mental_health_rf_model.pkl")
XGB_MODEL_PATH = os.path.join(MODELS_DIR, "mental_health_xgb_model.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")

# Feature Mappings
SIZE_MAPPING = {"1-5": 0, "6-25": 1, "26-100": 2, "100-500": 3, "101-500": 3, "501-1000": 4, "More than 1000": 5}
TECH_MAPPING = {"No": 0, "Yes": 1}
BENEFITS_MAPPING = {"No": 0, "Yes": 1, "Don't know": 2, "Not eligible for coverage / N/A": 3}
CARE_MAPPING = {"No": 0, "Yes": 1, "Not sure": 2}
DISCUSSION_MAPPING = {"No": 0, "Yes": 1, "Don't know": 2}
RESOURCES_MAPPING = {"No": 0, "Yes": 1, "Don't know": 2}
ANONYMITY_MAPPING = {"No": 0, "Yes": 1, "I don't know": 2, "Don't know": 2}
LEAVE_MAPPING = {"Very easy": 0, "Somewhat easy": 1, "Neither easy nor difficult": 2, "Somewhat difficult": 3, "Very difficult": 4, "Don't know": 5}
CONSEQUENCES_MAPPING = {"No": 0, "Yes": 1, "Maybe": 2}
SERIOUSNESS_MAPPING = {"No": 0, "Yes": 1, "I don't know": 2, "Don't know": 2}

CLASS_LABELS = {
    0: "No (Unlikely to seek treatment)",
    1: "Maybe (Undecided / Potential trend)",
    2: "Yes (Highly likely to seek treatment)"
}
