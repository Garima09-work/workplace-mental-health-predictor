import pickle
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb

from src.config import RF_MODEL_PATH, XGB_MODEL_PATH, ENCODER_PATH, MODELS_DIR
from src.preprocess import load_and_preprocess_data

def train_and_save_models():
    """
    Trains Random Forest and XGBoost Classifier models on the survey dataset
    and exports binary models and label encoder into models/ directory.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("[INFO] Loading and preprocessing survey dataset...")
    X, y = load_and_preprocess_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[INFO] Dataset split: {len(X_train)} train rows, {len(X_test)} test rows.")

    # 1. Label Encoder
    encoder = LabelEncoder()
    encoder.fit(["Maybe", "No", "Yes"])

    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(encoder, f)
    print(f"[SUCCESS] Saved Label Encoder to {ENCODER_PATH}")

    # 2. Random Forest Classifier
    print("[INFO] Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    rf_preds = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    print(f"[RESULT] Random Forest Accuracy: {rf_acc * 100:.2f}%")

    with open(RF_MODEL_PATH, "wb") as f:
        pickle.dump(rf_model, f)
    print(f"[SUCCESS] Saved Random Forest model to {RF_MODEL_PATH}")

    # 3. XGBoost Classifier / Regressor (Multi-class)
    print("[INFO] Training XGBoost Model...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        eval_metric='mlogloss'
    )
    xgb_model.fit(X_train, y_train)

    xgb_preds = xgb_model.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_preds)
    print(f"[RESULT] XGBoost Model Accuracy: {xgb_acc * 100:.2f}%")

    with open(XGB_MODEL_PATH, "wb") as f:
        pickle.dump(xgb_model, f)
    print(f"[SUCCESS] Saved XGBoost model to {XGB_MODEL_PATH}")

    print("\n[SUCCESS] All models trained and saved successfully!")

if __name__ == "__main__":
    train_and_save_models()
