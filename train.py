"""
FraudShield Advanced Training Module
Implements multiple models, hyperparameter tuning, and class imbalance handling
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedRandomForestClassifier
import optuna
import warnings
warnings.filterwarnings('ignore')

# Import utilities
import sys
sys.path.append(os.path.dirname(__file__))
from CreditCardFraudDetection.utils import get_or_create_dataset

class FraudDetectionTrainer:
    """
    Comprehensive fraud detection trainer with multiple models and techniques
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self.results = []
        self.best_model = None
        self.best_model_name = None
        
    def load_and_prepare_data(self, test_size=0.2):
        """Load dataset and prepare for training"""
        print("Loading dataset...")
        df = get_or_create_dataset()
        df = df.dropna()
        
        X = df.drop(columns=['Class'])
        y = df['Class']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale Time and Amount
        cols_to_scale = ['Time', 'Amount']
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        X_train_scaled[cols_to_scale] = self.scaler.fit_transform(X_train[cols_to_scale])
        X_test_scaled[cols_to_scale] = self.scaler.transform(X_test[cols_to_scale])
        
        print(f"Training set: {X_train_scaled.shape}, Test set: {X_test_scaled.shape}")
        print(f"Class distribution - Train: {y_train.value_counts().to_dict()}, Test: {y_test.value_counts().to_dict()}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def apply_class_imbalance_technique(self, X_train, y_train, technique='smote', sampling_ratio=0.1):
        """Apply various class imbalance handling techniques"""
        print(f"\nApplying {technique.upper()} for class imbalance...")
        
        if technique == 'smote':
            smote = SMOTE(sampling_strategy=sampling_ratio, random_state=self.random_state)
            X_res, y_res = smote.fit_resample(X_train, y_train)
            
        elif technique == 'undersampling':
            undersampler = RandomUnderSampler(sampling_strategy=sampling_ratio, random_state=self.random_state)
            X_res, y_res = undersampler.fit_resample(X_train, y_train)
            
        elif technique == 'hybrid':
            # First oversample minority, then undersample majority
            smote = SMOTE(sampling_strategy=0.05, random_state=self.random_state)
            X_temp, y_temp = smote.fit_resample(X_train, y_train)
            undersampler = RandomUnderSampler(sampling_strategy=0.3, random_state=self.random_state)
            X_res, y_res = undersampler.fit_resample(X_temp, y_temp)
            
        elif technique == 'none':
            X_res, y_res = X_train, y_train
            
        else:
            raise ValueError(f"Unknown technique: {technique}")
        
        print(f"After {technique}: {X_res.shape}, Class distribution: {y_res.value_counts().to_dict()}")
        return X_res, y_res
    
    def get_model_definitions(self):
        """Return dictionary of model configurations"""
        return {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, class_weight='balanced', random_state=self.random_state
            ),
            'Decision Tree': DecisionTreeClassifier(
                class_weight='balanced', random_state=self.random_state, max_depth=10
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, class_weight='balanced', random_state=self.random_state, n_jobs=-1
            ),
            'Balanced Random Forest': BalancedRandomForestClassifier(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=6, 
                random_state=self.random_state, eval_metric='logloss'
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=6,
                random_state=self.random_state, verbose=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=6, random_state=self.random_state
            ),
            'Neural Network (MLP)': MLPClassifier(
                hidden_layer_sizes=(100, 50), max_iter=500, random_state=self.random_state
            )
        }
    
    def train_single_model(self, model, X_train, y_train, X_test, y_test, model_name):
        """Train a single model and return metrics"""
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Get probabilities
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_prob = model.decision_function(X_test)
            y_prob = 1 / (1 + np.exp(-y_prob))  # sigmoid
        else:
            y_prob = y_pred
        
        # Calculate metrics
        metrics = {
            'Model': model_name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1': f1_score(y_test, y_pred, zero_division=0),
            'ROC AUC': roc_auc_score(y_test, y_prob)
        }
        
        print(f"{model_name} - Accuracy: {metrics['Accuracy']:.4f}, F1: {metrics['F1']:.4f}, ROC AUC: {metrics['ROC AUC']:.4f}")
        
        return model, metrics
    
    def train_all_models(self, X_train, y_train, X_test, y_test, imbalance_technique='smote'):
        """Train all models with specified imbalance technique"""
        print(f"\n{'='*60}")
        print(f"TRAINING ALL MODELS WITH {imbalance_technique.upper()}")
        print(f"{'='*60}")
        
        # Apply imbalance technique
        X_train_res, y_train_res = self.apply_class_imbalance_technique(
            X_train, y_train, technique=imbalance_technique
        )
        
        models = self.get_model_definitions()
        results = []
        
        for model_name, model in models.items():
            try:
                trained_model, metrics = self.train_single_model(
                    model, X_train_res, y_train_res, X_test, y_test, model_name
                )
                self.models[model_name] = trained_model
                metrics['Imbalance Technique'] = imbalance_technique
                results.append(metrics)
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                continue
        
        self.results = results
        
        # Find best model by F1 score
        if results:
            best_result = max(results, key=lambda x: x['F1'])
            self.best_model_name = best_result['Model']
            self.best_model = self.models[self.best_model_name]
            print(f"\n🏆 Best Model: {self.best_model_name} with F1: {best_result['F1']:.4f}")
        
        return results
    
    def hyperparameter_tuning(self, X_train, y_train, X_test, y_test, model_name='Random Forest', 
                             search_type='random', n_iter=20):
        """Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV"""
        print(f"\n{'='*60}")
        print(f"HYPERPARAMETER TUNING FOR {model_name.upper()}")
        print(f"{'='*60}")
        
        # Apply SMOTE for tuning
        X_train_res, y_train_res = self.apply_class_imbalance_technique(
            X_train, y_train, technique='smote'
        )
        
        # Define parameter grids
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'class_weight': ['balanced', 'balanced_subsample']
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            },
            'Logistic Regression': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'penalty': ['l2'],
                'solver': ['liblinear', 'lbfgs']
            }
        }
        
        if model_name not in param_grids:
            print(f"No parameter grid defined for {model_name}")
            return None
        
        # Get base model
        models = self.get_model_definitions()
        base_model = models.get(model_name)
        
        if base_model is None:
            print(f"Model {model_name} not available")
            return None
        
        # Store baseline performance
        baseline_model, baseline_metrics = self.train_single_model(
            base_model, X_train_res, y_train_res, X_test, y_test, f"{model_name} (Baseline)"
        )
        
        print(f"\nBaseline Performance: F1 = {baseline_metrics['F1']:.4f}")
        
        # Perform search
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        
        if search_type == 'grid':
            search = GridSearchCV(
                base_model, param_grids[model_name], cv=cv, 
                scoring='f1', n_jobs=-1, verbose=1
            )
        else:  # random search
            search = RandomizedSearchCV(
                base_model, param_grids[model_name], cv=cv,
                scoring='f1', n_iter=n_iter, n_jobs=-1, 
                random_state=self.random_state, verbose=1
            )
        
        print(f"\nPerforming {search_type.upper()} search...")
        search.fit(X_train_res, y_train_res)
        
        # Evaluate best model
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        if hasattr(best_model, 'predict_proba'):
            y_prob = best_model.predict_proba(X_test)[:, 1]
        else:
            y_prob = best_model.decision_function(X_test)
            y_prob = 1 / (1 + np.exp(-y_prob))
        
        tuned_metrics = {
            'Model': f"{model_name} (Tuned)",
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1': f1_score(y_test, y_pred, zero_division=0),
            'ROC AUC': roc_auc_score(y_test, y_prob),
            'Best Parameters': search.best_params_
        }
        
        print(f"\nTuned Performance: F1 = {tuned_metrics['F1']:.4f}")
        print(f"Best Parameters: {search.best_params_}")
        print(f"Improvement: {((tuned_metrics['F1'] - baseline_metrics['F1']) / baseline_metrics['F1'] * 100):.2f}%")
        
        return {
            'baseline': baseline_metrics,
            'tuned': tuned_metrics,
            'best_model': best_model,
            'improvement': tuned_metrics['F1'] - baseline_metrics['F1']
        }
    
    def cross_validation_analysis(self, X_train, y_train, model_name='Random Forest', cv=5):
        """Perform cross-validation analysis"""
        print(f"\n{'='*60}")
        print(f"CROSS-VALIDATION ANALYSIS FOR {model_name.upper()}")
        print(f"{'='*60}")
        
        X_train_res, y_train_res = self.apply_class_imbalance_technique(
            X_train, y_train, technique='smote'
        )
        
        models = self.get_model_definitions()
        model = models.get(model_name)
        
        if model is None:
            print(f"Model {model_name} not available")
            return None
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            model, X_train_res, y_train_res, cv=cv, 
            scoring='f1', n_jobs=-1
        )
        
        results = {
            'Model': model_name,
            'CV F1 Scores': cv_scores.tolist(),
            'Mean F1': cv_scores.mean(),
            'Std F1': cv_scores.std(),
            'Min F1': cv_scores.min(),
            'Max F1': cv_scores.max()
        }
        
        print(f"\nCross-Validation Results ({cv}-fold):")
        print(f"Mean F1: {results['Mean F1']:.4f} (+/- {results['Std F1']:.4f})")
        print(f"Range: [{results['Min F1']:.4f}, {results['Max F1']:.4f}]")
        
        return results
    
    def compare_imbalance_techniques(self, X_train, y_train, X_test, y_test):
        """Compare different class imbalance handling techniques"""
        print(f"\n{'='*60}")
        print("COMPARING CLASS IMBALANCE TECHNIQUES")
        print(f"{'='*60}")
        
        techniques = ['none', 'smote', 'undersampling', 'hybrid']
        comparison_results = []
        
        for technique in techniques:
            print(f"\n--- {technique.upper()} ---")
            results = self.train_all_models(X_train, y_train, X_test, y_test, technique)
            comparison_results.extend(results)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_results)
        
        # Save comparison
        os.makedirs('saved_models', exist_ok=True)
        comparison_df.to_csv('saved_models/imbalance_technique_comparison.csv', index=False)
        
        print(f"\nComparison results saved to saved_models/imbalance_technique_comparison.csv")
        
        return comparison_df
    
    def save_models(self, save_dir='saved_models'):
        """Save all trained models"""
        os.makedirs(save_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            filename = f"{model_name.replace(' ', '_').lower()}.pkl"
            filepath = os.path.join(save_dir, filename)
            joblib.dump(model, filepath)
            print(f"Saved {model_name} to {filepath}")
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(save_dir, 'scaler.pkl'))
        
        # Save results
        if self.results:
            results_df = pd.DataFrame(self.results)
            results_df.to_csv(os.path.join(save_dir, 'training_results.csv'), index=False)
            print(f"Saved training results to {save_dir}/training_results.csv")
    
    def generate_comparison_table(self):
        """Generate and return comparison table"""
        if not self.results:
            print("No training results available")
            return None
        
        df = pd.DataFrame(self.results)
        
        # Reorder columns
        cols = ['Model', 'Imbalance Technique', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC']
        df = df[cols]
        
        # Sort by F1 score
        df = df.sort_values('F1', ascending=False)
        
        return df

def main():
    """Main training pipeline"""
    print("="*60)
    print("FRAUDSHIELD ADVANCED TRAINING PIPELINE")
    print("="*60)
    
    trainer = FraudDetectionTrainer(random_state=42)
    
    # Load data
    X_train, X_test, y_train, y_test = trainer.load_and_prepare_data(test_size=0.2)
    
    # Task 1: Train multiple models with SMOTE
    print("\n" + "="*60)
    print("TASK 1: TRAINING MULTIPLE MODELS")
    print("="*60)
    results = trainer.train_all_models(X_train, y_train, X_test, y_test, imbalance_technique='smote')
    
    # Generate comparison table
    comparison_df = trainer.generate_comparison_table()
    print("\n" + "="*60)
    print("MODEL COMPARISON TABLE")
    print("="*60)
    print(comparison_df.to_string(index=False))
    
    # Task 2: Hyperparameter tuning
    print("\n" + "="*60)
    print("TASK 2: HYPERPARAMETER TUNING")
    print("="*60)
    tuning_results = trainer.hyperparameter_tuning(
        X_train, y_train, X_test, y_test, 
        model_name='Random Forest', search_type='random', n_iter=20
    )
    
    # Task 4: Compare imbalance techniques
    print("\n" + "="*60)
    print("TASK 4: COMPARING IMBALANCE TECHNIQUES")
    print("="*60)
    imbalance_comparison = trainer.compare_imbalance_techniques(X_train, y_train, X_test, y_test)
    
    # Task 10: Cross-validation analysis
    print("\n" + "="*60)
    print("TASK 10: CROSS-VALIDATION ANALYSIS")
    print("="*60)
    cv_results = trainer.cross_validation_analysis(X_train, y_train, model_name='Random Forest', cv=5)
    
    # Save all models and results
    print("\n" + "="*60)
    print("SAVING MODELS AND RESULTS")
    print("="*60)
    trainer.save_models()
    
    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)

if __name__ == "__main__":
    main()
