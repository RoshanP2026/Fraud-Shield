"""
FraudShield Explainable AI Module
Implements SHAP for model explainability and feature attribution
"""

import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ModelExplainer:
    """
    SHAP-based model explainer for fraud detection
    """
    
    def __init__(self, model, scaler=None, feature_names=None):
        """
        Initialize the explainer
        
        Args:
            model: Trained ML model
            scaler: Fitted scaler for data preprocessing
            feature_names: List of feature names
        """
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
    def create_explainer(self, X_background, explainer_type='auto'):
        """
        Create SHAP explainer based on model type
        
        Args:
            X_background: Background dataset for explainer
            explainer_type: 'auto', 'tree', 'kernel', 'deep', or 'linear'
        """
        print(f"Creating SHAP explainer (type: {explainer_type})...")
        
        if explainer_type == 'auto':
            # Auto-detect best explainer
            try:
                self.explainer = shap.TreeExplainer(self.model)
                print("Using TreeExplainer")
            except:
                try:
                    self.explainer = shap.LinearExplainer(self.model, X_background)
                    print("Using LinearExplainer")
                except:
                    self.explainer = shap.KernelExplainer(self.model.predict_proba, X_background)
                    print("Using KernelExplainer")
        elif explainer_type == 'tree':
            self.explainer = shap.TreeExplainer(self.model)
        elif explainer_type == 'linear':
            self.explainer = shap.LinearExplainer(self.model, X_background)
        elif explainer_type == 'kernel':
            self.explainer = shap.KernelExplainer(self.model.predict_proba, X_background)
        elif explainer_type == 'deep':
            self.explainer = shap.DeepExplainer(self.model, X_background)
        else:
            raise ValueError(f"Unknown explainer type: {explainer_type}")
        
        return self.explainer
    
    def compute_shap_values(self, X):
        """
        Compute SHAP values for given data
        
        Args:
            X: Input data (numpy array or pandas DataFrame)
        """
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer() first.")
        
        print("Computing SHAP values...")
        self.shap_values = self.explainer.shap_values(X)
        
        # Handle binary classification (return second class for fraud)
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]  # Use fraud class
        
        return self.shap_values
    
    def get_feature_importance(self, absolute=True):
        """
        Get global feature importance from SHAP values
        
        Args:
            absolute: If True, use absolute SHAP values
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values() first.")
        
        if absolute:
            importance = np.abs(self.shap_values).mean(axis=0)
        else:
            importance = self.shap_values.mean(axis=0)
        
        # Create DataFrame
        if self.feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importance))]
        else:
            feature_names = self.feature_names
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    def explain_single_prediction(self, X_instance, plot=True):
        """
        Explain a single prediction
        
        Args:
            X_instance: Single instance to explain
            plot: Whether to generate plot
        """
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer() first.")
        
        # Ensure X_instance is 2D
        if len(X_instance.shape) == 1:
            X_instance = X_instance.reshape(1, -1)
        
        # Compute SHAP values for this instance
        shap_values = self.explainer.shap_values(X_instance)
        
        # Handle binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Get base value (expected value)
        if hasattr(self.explainer, 'expected_value'):
            if isinstance(self.explainer.expected_value, list):
                base_value = self.explainer.expected_value[1]
            else:
                base_value = self.explainer.expected_value
        else:
            base_value = 0
        
        # Create feature attribution summary
        if self.feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(shap_values[0]))]
        else:
            feature_names = self.feature_names
        
        attribution = pd.DataFrame({
            'Feature': feature_names,
            'SHAP Value': shap_values[0],
            'Absolute Impact': np.abs(shap_values[0])
        }).sort_values('Absolute Impact', ascending=False)
        
        # Calculate percentage contribution
        total_impact = attribution['Absolute Impact'].sum()
        attribution['Contribution %'] = (attribution['Absolute Impact'] / total_impact * 100).round(2)
        
        if plot:
            self._plot_single_prediction(attribution, base_value)
        
        return attribution, base_value
    
    def _plot_single_prediction(self, attribution, base_value):
        """Create a waterfall-style plot for single prediction"""
        top_features = attribution.head(10)
        
        fig = go.Figure(go.Bar(
            x=top_features['SHAP Value'],
            y=top_features['Feature'],
            orientation='h',
            marker_color=['#F87171' if x > 0 else '#34D399' for x in top_features['SHAP Value']],
            text=top_features['Contribution %'].astype(str) + '%',
            textposition='outside'
        ))
        
        fig.update_layout(
            title='Feature Attribution for Prediction',
            xaxis_title='SHAP Value (Impact on Prediction)',
            yaxis_title='Features',
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            template='plotly_dark'
        )
        
        return fig
    
    def plot_summary_plot(self, save_path=None):
        """
        Create SHAP summary plot
        
        Args:
            save_path: Path to save the plot
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values() first.")
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(self.shap_values, feature_names=self.feature_names, show=False)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Summary plot saved to {save_path}")
        
        plt.close()
    
    def plot_feature_importance(self, top_n=15):
        """
        Create interactive feature importance plot
        
        Args:
            top_n: Number of top features to display
        """
        importance_df = self.get_feature_importance(absolute=True)
        top_features = importance_df.head(top_n)
        
        fig = go.Figure(go.Bar(
            x=top_features['Importance'],
            y=top_features['Feature'],
            orientation='h',
            marker_color='#38BDF8'
        ))
        
        fig.update_layout(
            title=f'Top {top_n} Feature Importance (SHAP Values)',
            xaxis_title='Mean |SHAP Value|',
            yaxis_title='Features',
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            template='plotly_dark'
        )
        
        return fig
    
    def plot_dependence_plot(self, feature_idx, interaction_idx='auto'):
        """
        Create SHAP dependence plot
        
        Args:
            feature_idx: Index of feature to plot
            interaction_idx: Index of interaction feature or 'auto'
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values() first.")
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_idx, self.shap_values, 
            feature_names=self.feature_names,
            interaction_idx=interaction_idx,
            show=False
        )
        
        plt.tight_layout()
        return plt.gcf()
    
    def generate_explanation_report(self, X_instance, prediction, probability):
        """
        Generate human-readable explanation for a prediction
        
        Args:
            X_instance: Input instance
            prediction: Model prediction (0 or 1)
            probability: Prediction probability
        """
        attribution, base_value = self.explain_single_prediction(X_instance, plot=False)
        
        top_positive = attribution[attribution['SHAP Value'] > 0].head(3)
        top_negative = attribution[attribution['SHAP Value'] < 0].head(3)
        
        explanation = f"""
        Prediction Explanation Report
        ==============================
        
        Prediction: {'FRAUDULENT' if prediction == 1 else 'LEGITIMATE'}
        Probability: {probability:.2%}
        
        Top Factors Increasing Fraud Risk:
        """
        
        for _, row in top_positive.iterrows():
            explanation += f"- {row['Feature']}: +{row['Contribution %']:.1f}% impact\n"
        
        explanation += "\nTop Factors Decreasing Fraud Risk:\n"
        
        for _, row in top_negative.iterrows():
            explanation += f"- {row['Feature']}: {row['Contribution %']:.1f}% impact\n"
        
        return explanation
    
    def save_explainer(self, save_path):
        """Save the explainer object"""
        joblib.dump({
            'explainer': self.explainer,
            'feature_names': self.feature_names,
            'scaler': self.scaler
        }, save_path)
        print(f"Explainer saved to {save_path}")
    
    def load_explainer(self, load_path):
        """Load a saved explainer"""
        data = joblib.load(load_path)
        self.explainer = data['explainer']
        self.feature_names = data['feature_names']
        self.scaler = data['scaler']
        print(f"Explainer loaded from {load_path}")


def create_shap_explainer_from_saved_model(model_path, scaler_path, X_background, feature_names=None):
    """
    Convenience function to create explainer from saved model
    
    Args:
        model_path: Path to saved model
        scaler_path: Path to saved scaler
        X_background: Background dataset
        feature_names: List of feature names
    """
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if scaler_path else None
    
    explainer = ModelExplainer(model, scaler, feature_names)
    explainer.create_explainer(X_background)
    
    return explainer


if __name__ == "__main__":
    # Example usage
    print("SHAP Explainability Module for FraudShield")
    print("This module provides SHAP-based model explainability")
