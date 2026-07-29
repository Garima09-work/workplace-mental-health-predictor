import pandas as pd
import numpy as np
from src.config import (
    DATA_RAW_PATH, SIZE_MAPPING, TECH_MAPPING, BENEFITS_MAPPING,
    CARE_MAPPING, DISCUSSION_MAPPING, RESOURCES_MAPPING, ANONYMITY_MAPPING,
    LEAVE_MAPPING, CONSEQUENCES_MAPPING, SERIOUSNESS_MAPPING
)

def load_and_preprocess_data(data_path=DATA_RAW_PATH):
    """
    Loads survey.csv, extracts 10 workplace features, and creates 3-class target variable.
    Returns X (DataFrame) and y (Series).
    """
    df = pd.read_csv(data_path)

    # Feature mapping
    df_clean = pd.DataFrame()
    df_clean['no_employees'] = df['no_employees'].map(SIZE_MAPPING).fillna(2)
    df_clean['tech_company'] = df['tech_company'].map(TECH_MAPPING).fillna(1)
    df_clean['benefits'] = df['benefits'].map(BENEFITS_MAPPING).fillna(2)
    df_clean['care_options'] = df['care_options'].map(CARE_MAPPING).fillna(2)
    df_clean['seek_help'] = df['seek_help'].map(DISCUSSION_MAPPING).fillna(2)
    df_clean['wellness_program'] = df['wellness_program'].map(RESOURCES_MAPPING).fillna(2)
    df_clean['anonymity'] = df['anonymity'].map(ANONYMITY_MAPPING).fillna(2)
    df_clean['leave'] = df['leave'].map(LEAVE_MAPPING).fillna(5)
    df_clean['mental_health_consequence'] = df['mental_health_consequence'].map(CONSEQUENCES_MAPPING).fillna(2)
    df_clean['mental_vs_physical'] = df['mental_vs_physical'].map(SERIOUSNESS_MAPPING).fillna(2)

    # Target variable construction (3 classes: 0 = No, 1 = Maybe, 2 = Yes)
    def construct_target(row):
        treatment = str(row['treatment']).strip()
        interfere = str(row['work_interfere']).strip()

        if treatment == 'Yes' and interfere in ['Often', 'Sometimes']:
            return 2
        elif treatment == 'Yes' or interfere in ['Rarely', 'Sometimes']:
            return 1
        else:
            return 0

    df_clean['target'] = df.apply(construct_target, axis=1)

    X = df_clean.drop(columns=['target'])
    y = df_clean['target']

    return X, y
