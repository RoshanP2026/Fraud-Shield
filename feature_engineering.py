"""
FraudShield Feature Engineering Module
Creates advanced features for fraud detection
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Advanced feature engineering for credit card fraud detection
    """
    
    def __init__(self, scaler=None):
        self.scaler = scaler or StandardScaler()
        self.feature_stats = {}
        
    def engineer_features(self, df, fit_stats=True):
        """
        Apply all feature engineering techniques
        
        Args:
            df: Input DataFrame with raw features
            fit_stats: Whether to fit statistics on this data
        """
        df = df.copy()
        
        # Ensure Time is in seconds from start
        if 'Time' in df.columns:
            df['Time'] = df['Time'].astype(float)
        
        # 1. Transaction Hour
        df = self._add_transaction_hour(df)
        
        # 2. Weekend Transaction
        df = self._add_weekend_flag(df)
        
        # 3. Transaction Velocity (transactions per hour)
        df = self._add_transaction_velocity(df, fit_stats=fit_stats)
        
        # 4. Large Transaction Flag
        df = self._add_large_transaction_flag(df, fit_stats=fit_stats)
        
        # 5. Amount Ratios
        df = self._add_amount_ratios(df, fit_stats=fit_stats)
        
        # 6. Transaction Amount Bins
        df = self._add_amount_bins(df)
        
        # 7. Time-based Features
        df = self._add_time_based_features(df)
        
        # 8. PCA Feature Interactions
        df = self._add_pca_interactions(df)
        
        # 9. Risk Score Features
        df = self._add_risk_scores(df, fit_stats=fit_stats)
        
        # 10. Distance-like Features (simulated)
        df = self._add_distance_features(df, fit_stats=fit_stats)
        
        return df
    
    def _add_transaction_hour(self, df):
        """Extract hour from Time (assuming Time is in seconds from start)"""
        if 'Time' not in df.columns:
            return df
        
        df['Transaction_Hour'] = (df['Time'] / 3600) % 24
        return df
    
    def _add_weekend_flag(self, df):
        """Add weekend flag (simulated based on hour patterns)"""
        if 'Transaction_Hour' not in df.columns:
            df = self._add_transaction_hour(df)
        
        # Simulate weekend based on transaction patterns
        # In real data, this would use actual dates
        df['Weekend_Transaction'] = ((df['Transaction_Hour'] >= 0) & (df['Transaction_Hour'] < 6)).astype(int)
        return df
    
    def _add_transaction_velocity(self, df, fit_stats=True, window_seconds=3600):
        """Calculate transaction velocity (transactions per hour)"""
        if 'Time' not in df.columns:
            return df
        
        df_sorted = df.sort_values('Time').copy()
        
        # Calculate rolling count of transactions in window
        df_sorted['Transaction_Velocity'] = df_sorted.groupby(
            pd.cut(df_sorted['Time'], bins=100)
        )['Time'].transform('count')
        
        # Normalize
        if fit_stats:
            mean_vel = df_sorted['Transaction_Velocity'].mean()
            std_vel = df_sorted['Transaction_Velocity'].std()
            self.feature_stats['velocity_mean'] = mean_vel
            self.feature_stats['velocity_std'] = std_vel
        else:
            mean_vel = self.feature_stats.get('velocity_mean', 1)
            std_vel = self.feature_stats.get('velocity_std', 1)
        
        df_sorted['Transaction_Velocity_Normalized'] = (
            (df_sorted['Transaction_Velocity'] - mean_vel) / (std_vel + 1e-8)
        )
        
        return df_sorted.sort_index()
    
    def _add_large_transaction_flag(self, df, fit_stats=True, threshold_percentile=95):
        """Add flag for unusually large transactions"""
        if 'Amount' not in df.columns:
            return df
        
        if fit_stats:
            threshold = df['Amount'].quantile(threshold_percentile / 100)
            self.feature_stats['large_amount_threshold'] = threshold
        else:
            threshold = self.feature_stats.get('large_amount_threshold', df['Amount'].quantile(0.95))
        
        df['Large_Transaction_Flag'] = (df['Amount'] > threshold).astype(int)
        return df
    
    def _add_amount_ratios(self, df, fit_stats=True):
        """Add amount-based ratio features"""
        if 'Amount' not in df.columns:
            return df
        
        if fit_stats:
            avg_amount = df['Amount'].mean()
            self.feature_stats['avg_amount'] = avg_amount
        else:
            avg_amount = self.feature_stats.get('avg_amount', 1)
        
        df['Amount_to_Average_Ratio'] = df['Amount'] / (avg_amount + 1e-8)
        
        # Log of amount
        df['Log_Amount'] = np.log1p(df['Amount'])
        
        return df
    
    def _add_amount_bins(self, df):
        """Bin transaction amounts into categories"""
        if 'Amount' not in df.columns:
            return df
        
        bins = [0, 10, 50, 100, 500, 1000, float('inf')]
        labels = ['Micro', 'Small', 'Medium', 'Large', 'Very Large', 'Ultra Large']
        
        df['Amount_Category'] = pd.cut(df['Amount'], bins=bins, labels=labels)
        df['Amount_Category_Code'] = df['Amount_Category'].cat.codes
        
        return df
    
    def _add_time_based_features(self, df):
        """Add time-based behavioral features"""
        if 'Time' not in df.columns:
            return df
        
        # Time of day categories
        df['Time_Period'] = pd.cut(
            df['Time'] / 3600 % 24,
            bins=[0, 6, 12, 18, 24],
            labels=['Night', 'Morning', 'Afternoon', 'Evening']
        )
        df['Time_Period_Code'] = df['Time_Period'].cat.codes
        
        # Seconds since last transaction (simulated)
        df_sorted = df.sort_values('Time').copy()
        df_sorted['Time_Since_Last'] = df_sorted['Time'].diff().fillna(df_sorted['Time'].iloc[0])
        df = df_sorted.sort_index()
        
        return df
    
    def _add_pca_interactions(self, df):
        """Add interaction features between key PCA components"""
        key_pca_features = ['V1', 'V2', 'V3', 'V4', 'V5', 'V10', 'V11', 'V12', 'V14', 'V17']
        
        available_features = [f for f in key_pca_features if f in df.columns]
        
        # Add key interactions
        if 'V14' in df.columns and 'V17' in df.columns:
            df['V14_V17_Interaction'] = df['V14'] * df['V17']
        
        if 'V4' in df.columns and 'V11' in df.columns:
            df['V4_V11_Interaction'] = df['V4'] * df['V11']
        
        if 'V14' in df.columns and 'Amount' in df.columns:
            df['V14_Amount_Interaction'] = df['V14'] * np.log1p(df['Amount'])
        
        # Sum of negatively correlated features
        negative_features = ['V14', 'V17', 'V12', 'V10']
        available_negative = [f for f in negative_features if f in df.columns]
        if available_negative:
            df['Negative_Feature_Sum'] = df[available_negative].sum(axis=1)
        
        # Sum of positively correlated features
        positive_features = ['V4', 'V11']
        available_positive = [f for f in positive_features if f in df.columns]
        if available_positive:
            df['Positive_Feature_Sum'] = df[available_positive].sum(axis=1)
        
        return df
    
    def _add_risk_scores(self, df, fit_stats=True):
        """Add composite risk score features"""
        # Risk based on amount
        if 'Amount' in df.columns:
            if fit_stats:
                amount_std = df['Amount'].std()
                self.feature_stats['amount_std'] = amount_std
            else:
                amount_std = self.feature_stats.get('amount_std', 1)
            
            df['Amount_Risk_Score'] = (df['Amount'] - df['Amount'].mean()) / (amount_std + 1e-8)
        
        # Risk based on PCA features
        risk_features = ['V14', 'V17', 'V12', 'V10', 'V4', 'V11']
        available_risk = [f for f in risk_features if f in df.columns]
        
        if available_risk:
            # Weighted risk score (negative features decrease risk, positive increase)
            weights = {'V14': -1, 'V17': -1, 'V12': -1, 'V10': -1, 'V4': 1, 'V11': 1}
            weighted_sum = sum(df.get(f, 0) * weights.get(f, 0) for f in available_risk)
            df['PCA_Risk_Score'] = weighted_sum
        
        return df
    
    def _add_distance_features(self, df, fit_stats=True):
        """Add distance-like features (simulated for fraud detection)"""
        # Simulate distance from home using PCA features
        if 'V1' in df.columns and 'V2' in df.columns:
            df['Distance_From_Home_Simulated'] = np.sqrt(df['V1']**2 + df['V2']**2)
        
        # Merchant risk (simulated using PCA features)
        if 'V3' in df.columns and 'V5' in df.columns:
            df['Merchant_Risk_Score'] = np.sqrt(df['V3']**2 + df['V5']**2)
        
        return df
    
    def get_feature_importance_names(self):
        """Return list of engineered feature names"""
        return [
            'Transaction_Hour',
            'Weekend_Transaction',
            'Transaction_Velocity_Normalized',
            'Large_Transaction_Flag',
            'Amount_to_Average_Ratio',
            'Log_Amount',
            'Amount_Category_Code',
            'Time_Period_Code',
            'Time_Since_Last',
            'V14_V17_Interaction',
            'V4_V11_Interaction',
            'V14_Amount_Interaction',
            'Negative_Feature_Sum',
            'Positive_Feature_Sum',
            'Amount_Risk_Score',
            'PCA_Risk_Score',
            'Distance_From_Home_Simulated',
            'Merchant_Risk_Score'
        ]
    
    def scale_features(self, df, features_to_scale=None):
        """Scale numerical features"""
        if features_to_scale is None:
            features_to_scale = self.get_feature_importance_names()
        
        available_features = [f for f in features_to_scale if f in df.columns]
        
        if available_features:
            df[available_features] = self.scaler.fit_transform(df[available_features])
        
        return df


def apply_feature_engineering_pipeline(df, fit_stats=True):
    """
    Convenience function to apply full feature engineering pipeline
    
    Args:
        df: Input DataFrame
        fit_stats: Whether to fit statistics on this data
    """
    engineer = FeatureEngineer()
    df_engineered = engineer.engineer_features(df, fit_stats=fit_stats)
    return df_engineered, engineer


if __name__ == "__main__":
    # Example usage
    print("Feature Engineering Module for FraudShield")
    print("This module provides advanced feature engineering capabilities")
