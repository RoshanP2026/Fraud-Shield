"""
Credit Card Fraud Detection System - Model Training & Tuning Script
Author: Senior ML Engineer & Python Developer
MSc Advanced Machine Learning Assignment Reference
"""

import os
import sys
import logging
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Fallback mechanism for XGBoost
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGB = False

# Import SMOTE from imbalanced-learn
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_ml_pipeline():
    logger.info("=" * 60)
    logger.info("STARTING CREDIT CARD FRAUD DETECTION MACHINE LEARNING PIPELINE")
    logger.info("=" * 60)
    
    # -------------------------------------------------------------------------
    # PART 1: LOAD & INSPECT DATASET
    # -------------------------------------------------------------------------
    logger.info("--- PART 1: Loading and Inspecting Dataset ---")
    
    # Generate/load a synthetic credit card dataset
    from utils import get_or_create_dataset, evaluate_classifier
    
    df = get_or_create_dataset()
    
    logger.info(f"Dataset Loaded Successfully! Shape: {df.shape}")
    logger.info("\n=== FIRST 5 ROWS ===")
    logger.info(df.head().to_string())
    
    logger.info("\n=== DATA STATISTICS ===")
    logger.info(df.describe().to_string())
    
    logger.info("\n=== MISSING VALUES ===")
    missing_vals = df.isnull().sum().sum()
    logger.info(f"Total Missing Values in Dataset: {missing_vals}")
    
    logger.info("\n=== DUPLICATE RECORDS ===")
    duplicates = df.duplicated().sum()
    logger.info(f"Total Duplicate Rows Found: {duplicates}")
    
    logger.info("\n=== CLASS DISTRIBUTION ===")
    class_counts = df['Class'].value_counts()
    class_pct = df['Class'].value_counts(normalize=True) * 100
    for cls, count in class_counts.items():
        logger.info(f"Class {cls}: {count} transactions ({class_pct[cls]:.4f}%)")
        
    # -------------------------------------------------------------------------
    # PART 3: DATA PREPROCESSING
    # -------------------------------------------------------------------------
    logger.info("\n--- PART 3: Preprocessing Data ---")
    
    # 1. Duplicate Removal
    if duplicates > 0:
        logger.info(f"Removing {duplicates} duplicate records...")
        df = df.drop_duplicates().reset_index(drop=True)
        logger.info(f"New shape after duplicate removal: {df.shape}")
    
    # 1.5. Missing Value Removal
    if missing_vals > 0:
        logger.info(f"Removing {missing_vals} missing values...")
        df = df.dropna().reset_index(drop=True)
        logger.info(f"New shape after missing value removal: {df.shape}")
        
    # 2. Extract features and target
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    # 3. Train-Test Split (80% Train, 20% Test, stratified to preserve rare class proportions)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    logger.info(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    logger.info(f"Train fraud count: {y_train.sum()} ({y_train.mean()*100:.4f}%)")
    logger.info(f"Test fraud count: {y_test.sum()} ({y_test.mean()*100:.4f}%)")
    
    # 4. Feature Scaling (Standardize Time and Amount; PCA features are already standardized)
    logger.info("Scaling Time and Amount features using StandardScaler...")
    scaler = StandardScaler()
    
    # Fit scaler on training data and transform both train and test
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    cols_to_scale = ['Time', 'Amount']
    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    
    # 5. Apply SMOTE to training set ONLY (To prevent validation leakage!)
    if HAS_SMOTE:
        logger.info("Applying SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance...")
        smote = SMOTE(sampling_strategy=0.1, random_state=42) # Resample minority to 10% of majority class
        X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
        logger.info(f"Resampled Training Set Shape: {X_train_res.shape}")
        logger.info(f"Resampled Training Class Distribution: Class 0: {(y_train_res == 0).sum()}, Class 1: {(y_train_res == 1).sum()}")
    else:
        logger.warning("SMOTE package (imbalanced-learn) not available. Proceeding with class weights instead.")
        X_train_res, y_train_res = X_train_scaled, y_train

    # -------------------------------------------------------------------------
    # PART 4: TRAIN 3 MODELS
    # -------------------------------------------------------------------------
    logger.info("\n--- PART 4: Training Classifiers ---")
    
    models = {}
    
    # Model 1: Logistic Regression
    logger.info("Training Model 1: Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr_model.fit(X_train_res, y_train_res)
    models['Logistic Regression'] = lr_model
    
    # Model 2: Random Forest
    logger.info("Training Model 2: Random Forest (MSc grade configuration)...")
    # Using small estimators for quick execution while maintaining structure
    rf_model = RandomForestClassifier(n_estimators=30, max_depth=12, class_weight='balanced_subsample', random_state=42, n_jobs=-1)
    rf_model.fit(X_train_res, y_train_res)
    models['Random Forest'] = rf_model
    
    # Model 3: XGBoost (or Fallback Gradient Boosting)
    if HAS_XGB:
        logger.info("Training Model 3: XGBoost Classifier...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=30,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=10 if not HAS_SMOTE else 1,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train_res, y_train_res)
        models['XGBoost'] = xgb_model
    else:
        logger.info("XGBoost not installed. Training Fallback Model 3: Gradient Boosting Classifier...")
        gb_model = GradientBoostingClassifier(
            n_estimators=30,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        gb_model.fit(X_train_res, y_train_res)
        models['Gradient Boosting'] = gb_model

    # -------------------------------------------------------------------------
    # PART 5: HYPERPARAMETER TUNING (GridSearchCV on best general classifier)
    # -------------------------------------------------------------------------
    logger.info("\n--- PART 5: Hyperparameter Tuning via GridSearchCV ---")
    
    # Tune Logistic Regression or a small Random Forest for computational efficiency
    logger.info("Tuning Logistic Regression parameters using GridSearchCV...")
    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0]
    }
    
    grid_search = GridSearchCV(
        estimator=LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        param_grid=param_grid,
        scoring='f1',
        cv=3,
        n_jobs=-1
    )
    
    grid_search.fit(X_train_res, y_train_res)
    best_lr = grid_search.best_estimator_
    logger.info(f"GridSearchCV Complete! Best parameters: {grid_search.best_params_}")
    logger.info(f"Best cross-validated F1-score: {grid_search.best_score_:.4f}")
    
    # Update Logistic Regression model in dictionary to the tuned version
    models['Tuned Logistic Regression'] = best_lr

    # -------------------------------------------------------------------------
    # PART 6: EVALUATION
    # -------------------------------------------------------------------------
    logger.info("\n--- PART 6 & 7: Model Evaluation & Overfitting Diagnostics ---")
    
    results = {}
    
    for name, model in models.items():
        logger.info(f"\n===== EVALUATING MODEL: {name} =====")
        
        # Test predictions
        y_test_pred = model.predict(X_test_scaled)
        if hasattr(model, "predict_proba"):
            y_test_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_test_prob = model.decision_function(X_test_scaled)
            
        # Train predictions
        y_train_pred = model.predict(X_train_res)
        if hasattr(model, "predict_proba"):
            y_train_prob = model.predict_proba(X_train_res)[:, 1]
        else:
            y_train_prob = model.decision_function(X_train_res)
            
        # Evaluation metrics
        train_metrics = evaluate_classifier(y_train_res, y_train_pred, y_train_prob)
        test_metrics = evaluate_classifier(y_test, y_test_pred, y_test_prob)
        
        results[name] = {
            'train': train_metrics,
            'test': test_metrics
        }
        
        logger.info(f"TRAIN Set - F1 Score: {train_metrics['f1']:.4f} | ROC AUC: {train_metrics['roc_auc']:.4f}")
        logger.info(f"TEST Set  - F1 Score: {test_metrics['f1']:.4f} | ROC AUC: {test_metrics['roc_auc']:.4f}")
        logger.info(f"TEST Set  - Recall: {test_metrics['recall']:.4f} | Precision: {test_metrics['precision']:.4f}")
        
        # Log difference to diagnose overfitting
        f1_diff = train_metrics['f1'] - test_metrics['f1']
        logger.info(f"F1 score discrepancy (Train vs Test): {f1_diff:.4f}")
        if f1_diff > 0.15:
            logger.warning(f"MODEL '{name}' EXHIBITS SIGNS OF OVERFITTING! (F1 discrepancy > 0.15)")
        else:
            logger.info(f"MODEL '{name}' generalizes well (F1 discrepancy <= 0.15).")
            
        # Log Confusion Matrix
        logger.info(f"Test Confusion Matrix:\n{test_metrics['confusion_matrix']}")
        
    # -------------------------------------------------------------------------
    # PART 8 & 9: SAVE THE BEST MODEL
    # -------------------------------------------------------------------------
    logger.info("\n--- PART 8: Saving Best Model & Scaler ---")
    
    # Determine the best model using Test F1 score as primary metric
    best_model_name = max(models.keys(), key=lambda name: results[name]['test']['f1'])
    best_model = models[best_model_name]
    
    logger.info(f"BEST MODEL CHOSEN: '{best_model_name}' with Test F1 Score of {results[best_model_name]['test']['f1']:.4f}")
    
    # Save directory
    save_dir = "CreditCardFraudDetection"
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = os.path.join(save_dir, "model.pkl")
    scaler_path = os.path.join(save_dir, "scaler.pkl")
    
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    
    logger.info(f"Best model saved successfully to: {model_path}")
    logger.info(f"StandardScaler saved successfully to: {scaler_path}")
    logger.info("Pipeline Execution Completed Successfully!")
    logger.info("=" * 60)

if __name__ == "__main__":
    run_ml_pipeline()
