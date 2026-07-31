import os
import json
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, balanced_accuracy_score,
    matthews_corrcoef, cohen_kappa_score
)
from sklearn.feature_selection import mutual_info_classif, SelectKBest, chi2
from sklearn.inspection import permutation_importance

# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE

# Hyperparameter Tuning
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Explainability
import shap

from sklearn.pipeline import Pipeline
from src.config import BEST_MODEL_PATH, METRICS_REPORT_PATH, MODELS_DIR, DATA_RAW_PATH, TARGET_COL
from src.preprocess import build_preprocessor, clean_raw_data, RawDataCleaner

def evaluate_model(model, X_test, y_test):
    """
    Evaluates a trained model on test data across 8 metrics:
    - Accuracy, Precision, Recall, F1 Score, ROC-AUC, Balanced Accuracy, MCC, Cohen's Kappa.
    """
    y_pred = model.predict(X_test)
    
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred
        
    metrics = {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "F1 Score": float(f1_score(y_test, y_pred, zero_division=0)),
        "ROC AUC": float(roc_auc_score(y_test, y_prob)),
        "Balanced Accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "MCC": float(matthews_corrcoef(y_test, y_pred)),
        "Cohen Kappa": float(cohen_kappa_score(y_test, y_pred)),
        "Confusion Matrix": confusion_matrix(y_test, y_pred).tolist()
    }
    
    return metrics

def tune_xgboost_optuna(X_train, y_train, n_trials=20):
    """
    Tunes XGBoost hyperparameters using Optuna over Stratified 5-Fold CV.
    """
    print("   [OPTUNA] Tuning XGBoost Hyperparameters...")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        model = xgb.XGBClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
        return scores.mean()
        
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    best_params['random_state'] = 42
    best_params['eval_metric'] = 'logloss'
    
    best_xgb = xgb.XGBClassifier(**best_params)
    best_xgb.fit(X_train, y_train)
    return best_xgb, best_params

def tune_random_forest(X_train, y_train):
    """
    Tunes Random Forest using RandomizedSearchCV over Stratified 5-Fold CV.
    """
    print("   [SEARCH] Tuning Random Forest Hyperparameters...")
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced', None]
    }
    rf = RandomForestClassifier(random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=15, cv=cv, scoring='f1', random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_

def tune_catboost(X_train, y_train):
    """
    Tunes CatBoost Classifier using RandomizedSearchCV over Stratified 5-Fold CV.
    """
    print("   [SEARCH] Tuning CatBoost Hyperparameters...")
    param_dist = {
        'iterations': [100, 200, 300],
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'l2_leaf_reg': [1, 3, 5]
    }
    cb_model = cb.CatBoostClassifier(random_state=42, verbose=0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(cb_model, param_distributions=param_dist, n_iter=10, cv=cv, scoring='f1', random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_

def run_feature_analysis(X_train_trans, y_train, feature_names):
    """
    Computes Mutual Information scores across preprocessed features.
    """
    print("\n--- Feature Importance & Selection Analysis ---")
    mi_scores = mutual_info_classif(X_train_trans, y_train, random_state=42)
    mi_df = pd.DataFrame({"Feature": feature_names, "Mutual Information": mi_scores})
    mi_df = mi_df.sort_values(by="Mutual Information", ascending=False).reset_index(drop=True)
    
    print("\nTop 10 Features by Mutual Information:")
    print(mi_df.head(10).to_string(index=False))
    return mi_df

def run_shap_analysis(model, X_train_trans, feature_names):
    """
    Computes SHAP feature importance for model explainability.
    """
    print("\n--- SHAP Model Explainability Analysis ---")
    try:
        explainer = shap.Explainer(model, X_train_trans)
        shap_values = explainer(X_train_trans)
        
        # Calculate mean absolute SHAP values per feature
        if hasattr(shap_values, "values"):
            vals = np.abs(shap_values.values)
            if len(vals.shape) == 3: # multi-output fallback
                vals = vals[:, :, 1]
            mean_shap = np.mean(vals, axis=0)
        else:
            mean_shap = np.abs(shap_values).mean(axis=0)
            
        shap_df = pd.DataFrame({"Feature": feature_names, "Mean |SHAP Value|": mean_shap})
        shap_df = shap_df.sort_values(by="Mean |SHAP Value|", ascending=False).reset_index(drop=True)
        
        print("\nTop 10 Most Influential Features (SHAP):")
        print(shap_df.head(10).to_string(index=False))
        return shap_df
    except Exception as e:
        print(f"   [NOTE] SHAP computation note: {e}")
        return None

def train_pipeline():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print("[1/6] Loading Raw Dataset...")
    raw_df = pd.read_csv(DATA_RAW_PATH)
    
    # Stratified Train-Test Split ON RAW DATA (Preventing data leakage)
    X_raw = raw_df.drop(columns=[TARGET_COL])
    y_raw = raw_df[TARGET_COL].map({"Yes": 1, "No": 0})
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.20, random_state=42, stratify=y_raw
    )
    print(f"      Train Set: {len(X_train_raw)} samples | Test Set: {len(X_test_raw)} samples")
    
    # Fit Cleaner + Preprocessor ONLY on X_train_raw
    print("[2/6] Cleaning Data & Fitting ColumnTransformer on Train Data...")
    cleaner = RawDataCleaner()
    X_train_cleaned = cleaner.fit_transform(X_train_raw)
    X_test_cleaned = cleaner.transform(X_test_raw)
    
    preprocessor = build_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train_cleaned)
    X_test_trans = preprocessor.transform(X_test_cleaned)
    
    # Extract feature names after transformer
    num_cols = preprocessor.transformers_[0][2]
    ord_cols = preprocessor.transformers_[1][2]
    nom_encoder = preprocessor.transformers_[2][1].named_steps['onehot']
    nom_cols = nom_encoder.get_feature_names_out(preprocessor.transformers_[2][2]).tolist()
    feature_names = list(num_cols) + list(ord_cols) + nom_cols
    
    print(f"      Extracted {len(feature_names)} features after encoding & scaling.")
    
    # 3. Feature Selection / Analysis
    run_feature_analysis(X_train_trans, y_train, feature_names)
    
    # 4. Model Suite Definition
    print("\n[3/6] Benchmarking Machine Learning Algorithms...")
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, eval_metric='logloss'),
        "CatBoost": cb.CatBoostClassifier(iterations=200, depth=6, learning_rate=0.05, random_state=42, verbose=0),
        "LightGBM": lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=150, max_depth=10, random_state=42)
    }
    
    baseline_results = {}
    for name, model in models.items():
        model.fit(X_train_trans, y_train)
        metrics = evaluate_model(model, X_test_trans, y_test)
        baseline_results[name] = metrics
        print(f"   [MODEL] {name:20s} | Acc: {metrics['Accuracy']*100:.2f}% | F1: {metrics['F1 Score']:.4f} | ROC-AUC: {metrics['ROC AUC']:.4f} | MCC: {metrics['MCC']:.4f}")
        
    # 5. Hyperparameter Tuning
    print("\n[4/6] Performing Advanced Hyperparameter Tuning (Optuna & RandomizedSearch)...")
    tuned_xgb, xgb_params = tune_xgboost_optuna(X_train_trans, y_train, n_trials=25)
    xgb_metrics = evaluate_model(tuned_xgb, X_test_trans, y_test)
    print(f"   [TUNED] Tuned XGBoost           | Acc: {xgb_metrics['Accuracy']*100:.2f}% | F1: {xgb_metrics['F1 Score']:.4f} | ROC-AUC: {xgb_metrics['ROC AUC']:.4f} | MCC: {xgb_metrics['MCC']:.4f}")
    
    tuned_rf, rf_params = tune_random_forest(X_train_trans, y_train)
    rf_metrics = evaluate_model(tuned_rf, X_test_trans, y_test)
    print(f"   [TUNED] Tuned Random Forest     | Acc: {rf_metrics['Accuracy']*100:.2f}% | F1: {rf_metrics['F1 Score']:.4f} | ROC-AUC: {rf_metrics['ROC AUC']:.4f} | MCC: {rf_metrics['MCC']:.4f}")
    
    tuned_cb, cb_params = tune_catboost(X_train_trans, y_train)
    cb_metrics = evaluate_model(tuned_cb, X_test_trans, y_test)
    print(f"   [TUNED] Tuned CatBoost          | Acc: {cb_metrics['Accuracy']*100:.2f}% | F1: {cb_metrics['F1 Score']:.4f} | ROC-AUC: {cb_metrics['ROC AUC']:.4f} | MCC: {cb_metrics['MCC']:.4f}")
    
    # Compare all candidates
    all_evaluated = {
        **baseline_results,
        "Tuned XGBoost": xgb_metrics,
        "Tuned Random Forest": rf_metrics,
        "Tuned CatBoost": cb_metrics
    }
    
    candidate_models = {
        **models,
        "Tuned XGBoost": tuned_xgb,
        "Tuned Random Forest": tuned_rf,
        "Tuned CatBoost": tuned_cb
    }
    
    # Select best model based on F1-Score & ROC-AUC
    best_model_name = max(all_evaluated, key=lambda k: (all_evaluated[k]['F1 Score'], all_evaluated[k]['ROC AUC']))
    best_estimator = candidate_models[best_model_name]
    best_metrics = all_evaluated[best_model_name]
    
    print(f"\n[WINNER] Best Selected Model: {best_model_name}")
    print(f"   Accuracy:          {best_metrics['Accuracy']*100:.2f}%")
    print(f"   F1 Score:          {best_metrics['F1 Score']:.4f}")
    print(f"   ROC AUC:           {best_metrics['ROC AUC']:.4f}")
    print(f"   Balanced Accuracy: {best_metrics['Balanced Accuracy']:.4f}")
    print(f"   MCC:               {best_metrics['MCC']:.4f}")
    print(f"   Cohen's Kappa:     {best_metrics['Cohen Kappa']:.4f}")
    
    # 6. SHAP Explainability on Best Model
    run_shap_analysis(best_estimator, X_train_trans, feature_names)
    
    # 7. Construct Full End-to-End Raw Inference Pipeline & Export
    print("\n[5/6] Exporting End-to-End Prediction Pipeline...")
    full_pipeline = Pipeline(steps=[
        ("cleaner", cleaner),
        ("preprocessor", preprocessor),
        ("classifier", best_estimator)
    ])
    
    joblib.dump(full_pipeline, BEST_MODEL_PATH)
    print(f"   [SUCCESS] Full end-to-end pipeline saved to: {BEST_MODEL_PATH}")
    
    # Save JSON metrics report
    report_data = {
        "best_model_name": best_model_name,
        "best_metrics": best_metrics,
        "all_model_metrics": all_evaluated
    }
    with open(METRICS_REPORT_PATH, "w") as f:
        json.dump(report_data, f, indent=4)
    print(f"   [SUCCESS] Metrics report saved to: {METRICS_REPORT_PATH}")

if __name__ == "__main__":
    train_pipeline()
