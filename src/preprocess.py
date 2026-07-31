import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.config import (
    DATA_RAW_PATH, TARGET_COL, DROP_COLS, NUMERIC_COLS,
    ORDINAL_COLS, ORDINAL_CATEGORIES, NOMINAL_COLS
)

def standardize_gender(val):
    if pd.isna(val):
        return "Other"
    v = str(val).strip().lower()
    
    males = {
        'm', 'male', 'male-ish', 'maile', 'cis male', 'something kinda male?',
        'mal', 'male (cis)', 'make', 'guy (-ish) ^_^', 'male leaning androgynous',
        'man', 'msle', 'mail', 'malr', 'cis man', 'ostensibly male, unsure what that really means'
    }
    females = {
        'f', 'female', 'cis female', 'woman', 'femake', 'cis-female/femme',
        'female (cis)', 'femail'
    }
    non_binary = {
        'trans-female', 'queer/she/they', 'non-binary', 'enby', 'fluid', 'genderqueer',
        'androgyne', 'agender', 'trans woman', 'neuter', 'female (trans)', 'queer', 'p'
    }
    
    if v in males:
        return "Male"
    elif v in females:
        return "Female"
    elif v in non_binary:
        return "Non-binary"
    else:
        return "Other"

def clean_raw_data(df):
    """
    Cleans raw OSMI survey DataFrame:
    - Removes duplicate records.
    - Standardizes Gender.
    - Imputes/cleans Age outliers (18 to 100 range).
    - Handles state logic (retains state if Country == 'United States', else 'Not Applicable').
    - Groups low-cardinality rare countries.
    - Imputes missing values in work_interfere and self_employed.
    - Engineered domain composite scores.
    """
    df = df.copy()
    
    # 1. Remove duplicate records if fitting full dataset
    if len(df) > 1:
        df = df.drop_duplicates().reset_index(drop=True)
    
    # 2. Standardize Gender
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].apply(standardize_gender)
    
    # 3. Clean Age (valid range 18 to 100, replace outliers with median)
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
        valid_ages = df[(df["Age"] >= 18) & (df["Age"] <= 100)]["Age"]
        median_age = valid_ages.median() if len(valid_ages) > 0 else 32.0
        df["Age"] = df["Age"].apply(lambda x: x if pd.notna(x) and (18 <= x <= 100) else median_age)
    
    # 4. Clean State column based on Country requirement
    def clean_state(row):
        country = str(row.get("Country", "")).strip() if pd.notna(row.get("Country")) else ""
        st = row.get("state", "")
        if country == "United States":
            return str(st).strip() if pd.notna(st) and str(st).strip() != "" else "Unknown"
        return "Not Applicable"
    
    if "Country" in df.columns and "state" in df.columns:
        df["state"] = df.apply(clean_state, axis=1)
    
    # 5. Clean Country (group rare countries with count < 5 into 'Other')
    if "Country" in df.columns:
        country_counts = df["Country"].value_counts()
        frequent_countries = country_counts[country_counts >= 5].index
        df["Country"] = df["Country"].apply(lambda c: c if c in frequent_countries else "Other")
    
    # 6. Missing value handling
    if "work_interfere" in df.columns:
        df["work_interfere"] = df["work_interfere"].fillna("Don't know")
    if "self_employed" in df.columns:
        df["self_employed"] = df["self_employed"].fillna("No")
    
    # 7. Feature Engineering (Composite Scores)
    # Company support score
    benefits_score = df.get("benefits", pd.Series()).map({"Yes": 2, "Don't know": 1, "No": 0, "Not eligible for coverage / N/A": 0}).fillna(0)
    care_score = df.get("care_options", pd.Series()).map({"Yes": 2, "Not sure": 1, "No": 0}).fillna(0)
    wellness_score = df.get("wellness_program", pd.Series()).map({"Yes": 2, "Don't know": 1, "No": 0}).fillna(0)
    help_score = df.get("seek_help", pd.Series()).map({"Yes": 2, "Don't know": 1, "No": 0}).fillna(0)
    df["company_support_score"] = benefits_score + care_score + wellness_score + help_score
    
    # Consequence concern score
    mh_conseq = df.get("mental_health_consequence", pd.Series()).map({"Yes": 2, "Maybe": 1, "No": 0}).fillna(0)
    ph_conseq = df.get("phys_health_consequence", pd.Series()).map({"Yes": 2, "Maybe": 1, "No": 0}).fillna(0)
    obs_conseq = df.get("obs_consequence", pd.Series()).map({"Yes": 2, "No": 0}).fillna(0)
    df["consequence_concern_score"] = mh_conseq + ph_conseq + obs_conseq
    
    # Workplace comfort score
    coworkers_score = df.get("coworkers", pd.Series()).map({"Yes": 2, "Some of them": 1, "No": 0}).fillna(0)
    supervisor_score = df.get("supervisor", pd.Series()).map({"Yes": 2, "Some of them": 1, "No": 0}).fillna(0)
    anonymity_score = df.get("anonymity", pd.Series()).map({"Yes": 2, "Don't know": 1, "I don't know": 1, "No": 0}).fillna(0)
    leave_score = df.get("leave", pd.Series()).map({
        "Very easy": 2, "Somewhat easy": 1, "Neither easy nor difficult": 0,
        "Don't know": 0, "Somewhat difficult": -1, "Very difficult": -2
    }).fillna(0)
    df["workplace_comfort_score"] = coworkers_score + supervisor_score + anonymity_score + leave_score
    
    # 8. Standardize Target variable if present
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})
    
    # 9. Drop uninformative / free-text columns
    df = df.drop(columns=[col for col in DROP_COLS if col in df.columns])
    
    return df

class RawDataCleaner(BaseEstimator, TransformerMixin):
    """
    Scikit-learn Transformer wrapper for clean_raw_data to allow seamless pipeline integration.
    """
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        return clean_raw_data(X)

def build_preprocessor():
    """
    Constructs scikit-learn ColumnTransformer:
    - NUMERIC_COLS: Median Imputer + StandardScaler
    - ORDINAL_COLS: Most Frequent Imputer + OrdinalEncoder
    - NOMINAL_COLS: Most Frequent Imputer + OneHotEncoder
    Ensures zero data leakage as preprocessor is fitted only on training data.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    ordinal_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(categories=ORDINAL_CATEGORIES, handle_unknown="use_encoded_value", unknown_value=-1))
    ])
    
    nominal_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLS),
            ("ord", ordinal_transformer, ORDINAL_COLS),
            ("nom", nominal_transformer, NOMINAL_COLS)
        ],
        remainder="drop"
    )
    
    return preprocessor

def get_preprocessed_data(data_path=DATA_RAW_PATH):
    """
    Loads raw CSV, cleans dataset, and returns X (features) and y (target).
    """
    raw_df = pd.read_csv(data_path)
    clean_df = clean_raw_data(raw_df)
    
    X = clean_df.drop(columns=[TARGET_COL])
    y = clean_df[TARGET_COL]
    
    return X, y
