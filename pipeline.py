"""
FraudShield ML Pipeline Module
End-to-end ML pipeline for fraud detection
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from feature_engineering import FeatureEngineer
from train import FraudDetectionTrainer

class MLPipeline:
    """
    Complete ML pipeline for fraud detection
    """
    
    def __init__(self, config=None):
        """
        Initialize the pipeline
        
        Args:
            config: Dictionary with pipeline configuration
        """
        self.config = config or {}
        self.feature_engineer = FeatureEngineer()
        self.trainer = FraudDetectionTrainer()
        self.scaler = StandardScaler()
        self.model = None
        self.pipeline_state = {}
        
    def load_data(self, filepath):
        """Load raw data"""
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def clean_data(self, df):
        """Step 1: Data Cleaning"""
        print("Step 1: Data Cleaning...")
        
        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed_duplicates = initial_rows - len(df)
        
        # Handle missing values
        missing_before = df.isnull().sum().sum()
        df = df.dropna()
        missing_after = df.isnull().sum().sum()
        
        # Remove negative amounts (invalid)
        if 'Amount' in df.columns:
            df = df[df['Amount'] >= 0]
        
        print(f"  - Removed {removed_duplicates} duplicates")
        print(f"  - Handled {missing_before} missing values")
        print(f"  - Final dataset: {df.shape}")
        
        self.pipeline_state['cleaning'] = {
            'initial_rows': initial_rows,
            'final_rows': len(df),
            'duplicates_removed': removed_duplicates,
            'missing_values_handled': missing_before
        }
        
        return df
    
    def encode_data(self, df):
        """Step 2: Encoding (if needed)"""
        print("Step 2: Encoding...")
        
        # Most features are already numerical (PCA features)
        # Handle categorical features if any
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col != 'Class':  # Don't encode target
                df[col] = pd.factorize(df[col])[0]
                print(f"  - Encoded {col}")
        
        return df
    
    def scale_features(self, df, fit=True):
        """Step 3: Feature Scaling"""
        print("Step 3: Feature Scaling...")
        
        # Identify numerical columns to scale
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != 'Class']
        
        if fit:
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
            print(f"  - Fitted and scaled {len(numerical_cols)} numerical features")
        else:
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
            print(f"  - Transformed {len(numerical_cols)} numerical features")
        
        return df
    
    def engineer_features(self, df, fit=True):
        """Step 4: Feature Engineering"""
        print("Step 4: Feature Engineering...")
        
        df_engineered = self.feature_engineer.engineer_features(df, fit_stats=fit)
        
        new_features = len(df_engineered.columns) - len(df.columns)
        print(f"  - Added {new_features} engineered features")
        print(f"  - Total features: {len(df_engineered.columns)}")
        
        return df_engineered
    
    def split_data(self, df, test_size=0.2, stratify=True):
        """Step 5: Train/Test Split"""
        print("Step 5: Train/Test Split...")
        
        X = df.drop(columns=['Class'])
        y = df['Class']
        
        if stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        print(f"  - Training set: {X_train.shape}")
        print(f"  - Test set: {X_test.shape}")
        print(f"  - Train fraud rate: {y_train.mean():.4f}")
        print(f"  - Test fraud rate: {y_test.mean():.4f}")
        
        self.pipeline_state['split'] = {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'train_fraud_rate': y_train.mean(),
            'test_fraud_rate': y_test.mean()
        }
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train, model_name='Random Forest', 
                    imbalance_technique='smote'):
        """Step 6: Model Training"""
        print(f"Step 6: Training {model_name} with {imbalance_technique}...")
        
        # Apply imbalance technique
        X_train_res, y_train_res = self.trainer.apply_class_imbalance_technique(
            X_train, y_train, technique=imbalance_technique
        )
        
        # Get model
        models = self.trainer.get_model_definitions()
        model = models.get(model_name)
        
        if model is None:
            raise ValueError(f"Model {model_name} not available")
        
        # Train
        model.fit(X_train_res, y_train_res)
        self.model = model
        
        print(f"  - Trained {model_name}")
        
        return model
    
    def evaluate_model(self, X_test, y_test):
        """Step 7: Model Evaluation"""
        print("Step 7: Model Evaluation...")
        
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        
        # Get probabilities
        if hasattr(self.model, 'predict_proba'):
            y_prob = self.model.predict_proba(X_test)[:, 1]
        elif hasattr(self.model, 'decision_function'):
            y_prob = self.model.decision_function(X_test)
            y_prob = 1 / (1 + np.exp(-y_prob))
        else:
            y_prob = y_pred
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_prob)
        }
        
        self.pipeline_state['evaluation'] = metrics
        
        print(f"  - Accuracy: {metrics['accuracy']:.4f}")
        print(f"  - Precision: {metrics['precision']:.4f}")
        print(f"  - Recall: {metrics['recall']:.4f}")
        print(f"  - F1-Score: {metrics['f1']:.4f}")
        print(f"  - ROC-AUC: {metrics['roc_auc']:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return metrics, y_pred, y_prob
    
    def deploy_model(self, save_dir='saved_models'):
        """Step 8: Model Deployment"""
        print("Step 8: Model Deployment...")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        if self.model:
            joblib.dump(self.model, os.path.join(save_dir, 'deployed_model.pkl'))
            print(f"  - Saved model to {save_dir}/deployed_model.pkl")
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(save_dir, 'deployed_scaler.pkl'))
        print(f"  - Saved scaler to {save_dir}/deployed_scaler.pkl")
        
        # Save feature engineer
        joblib.dump(self.feature_engineer, os.path.join(save_dir, 'feature_engineer.pkl'))
        print(f"  - Saved feature engineer to {save_dir}/feature_engineer.pkl")
        
        # Save pipeline state
        joblib.dump(self.pipeline_state, os.path.join(save_dir, 'pipeline_state.pkl'))
        print(f"  - Saved pipeline state to {save_dir}/pipeline_state.pkl")
        
        return True
    
    def run_full_pipeline(self, data_path, model_name='Random Forest', 
                         imbalance_technique='smote', save_dir='saved_models'):
        """
        Run the complete pipeline end-to-end
        
        Pipeline Stages:
        1. Raw Data
        2. Cleaning
        3. Encoding
        4. Scaling
        5. Feature Engineering
        6. Training
        7. Evaluation
        8. Deployment
        """
        print("="*60)
        print("FRAUDSHIELD ML PIPELINE")
        print("="*60)
        
        # Load data
        df = self.load_data(data_path)
        
        # Step 1: Cleaning
        df = self.clean_data(df)
        
        # Step 2: Encoding
        df = self.encode_data(df)
        
        # Step 3: Scaling
        df = self.scale_features(df, fit=True)
        
        # Step 4: Feature Engineering
        df = self.engineer_features(df, fit=True)
        
        # Step 5: Split
        X_train, X_test, y_train, y_test = self.split_data(df)
        
        # Step 6: Training
        self.train_model(X_train, y_train, model_name, imbalance_technique)
        
        # Step 7: Evaluation
        metrics, y_pred, y_prob = self.evaluate_model(X_test, y_test)
        
        # Step 8: Deployment
        self.deploy_model(save_dir)
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
        
        return {
            'pipeline_state': self.pipeline_state,
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_prob
        }
    
    def predict_single(self, transaction_data):
        """
        Make prediction on a single transaction
        
        Args:
            transaction_data: Dictionary or DataFrame with transaction features
        """
        if self.model is None:
            raise ValueError("Model not loaded. Train or load a model first.")
        
        # Convert to DataFrame if needed
        if isinstance(transaction_data, dict):
            df = pd.DataFrame([transaction_data])
        else:
            df = transaction_data.copy()
        
        # Apply feature engineering
        df = self.feature_engineer.engineer_features(df, fit_stats=False)
        
        # Scale
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        # Predict
        prediction = self.model.predict(df)[0]
        
        # Get probability
        if hasattr(self.model, 'predict_proba'):
            probability = self.model.predict_proba(df)[0, 1]
        elif hasattr(self.model, 'decision_function'):
            prob = self.model.decision_function(df)[0]
            probability = 1 / (1 + np.exp(-prob))
        else:
            probability = float(prediction)
        
        return {
            'prediction': int(prediction),
            'probability': float(probability),
            'is_fraud': bool(prediction == 1)
        }


def load_deployed_pipeline(save_dir='saved_models'):
    """
    Load a deployed pipeline
    
    Args:
        save_dir: Directory containing saved pipeline components
    """
    print(f"Loading deployed pipeline from {save_dir}...")
    
    pipeline = MLPipeline()
    
    # Load components
    pipeline.model = joblib.load(os.path.join(save_dir, 'deployed_model.pkl'))
    pipeline.scaler = joblib.load(os.path.join(save_dir, 'deployed_scaler.pkl'))
    pipeline.feature_engineer = joblib.load(os.path.join(save_dir, 'feature_engineer.pkl'))
    pipeline.pipeline_state = joblib.load(os.path.join(save_dir, 'pipeline_state.pkl'))
    
    print("Pipeline loaded successfully")
    
    return pipeline


if __name__ == "__main__":
    # Example usage
    print("ML Pipeline Module for FraudShield")
    print("This module provides end-to-end ML pipeline functionality")
