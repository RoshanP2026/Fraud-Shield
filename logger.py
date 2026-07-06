"""
FraudShield Logging Module
Handles prediction logging for deployment and monitoring
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List
import csv

class PredictionLogger:
    """
    Logs predictions with timestamps, probabilities, and features
    """
    
    def __init__(self, log_dir='logs', log_file='predictions.csv'):
        """
        Initialize the logger
        
        Args:
            log_dir: Directory to store log files
            log_file: Name of the log file
        """
        self.log_dir = log_dir
        self.log_file = log_file
        self.log_path = os.path.join(log_dir, log_file)
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize log file with headers if it doesn't exist
        if not os.path.exists(self.log_path):
            self._initialize_log_file()
    
    def _initialize_log_file(self):
        """Create log file with headers"""
        headers = [
            'timestamp',
            'prediction',
            'probability',
            'is_fraud',
            'amount',
            'time',
            'v14',
            'v17',
            'v4',
            'v11',
            'model_version',
            'processing_time_ms'
        ]
        
        with open(self.log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def log_prediction(self, prediction_data: Dict[str, Any], 
                       processing_time_ms: float = None,
                       model_version: str = "1.0"):
        """
        Log a single prediction
        
        Args:
            prediction_data: Dictionary containing prediction results
            processing_time_ms: Time taken for prediction in milliseconds
            model_version: Version of the model used
        """
        timestamp = datetime.now().isoformat()
        
        # Extract key features
        features = prediction_data.get('features', {})
        
        log_entry = {
            'timestamp': timestamp,
            'prediction': prediction_data.get('prediction', -1),
            'probability': prediction_data.get('probability', 0.0),
            'is_fraud': prediction_data.get('is_fraud', False),
            'amount': features.get('Amount', 0),
            'time': features.get('Time', 0),
            'v14': features.get('V14', 0),
            'v17': features.get('V17', 0),
            'v4': features.get('V4', 0),
            'v11': features.get('V11', 0),
            'model_version': model_version,
            'processing_time_ms': processing_time_ms or 0
        }
        
        # Append to log file
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_entry.keys())
            writer.writerow(log_entry)
    
    def log_batch_predictions(self, predictions: List[Dict[str, Any]], 
                            processing_time_ms: float = None,
                            model_version: str = "1.0"):
        """
        Log multiple predictions at once
        
        Args:
            predictions: List of prediction dictionaries
            processing_time_ms: Total time for all predictions
            model_version: Version of the model used
        """
        for pred in predictions:
            self.log_prediction(pred, processing_time_ms / len(predictions), model_version)
    
    def get_recent_predictions(self, n: int = 100) -> pd.DataFrame:
        """
        Get the most recent predictions
        
        Args:
            n: Number of recent predictions to retrieve
        """
        if not os.path.exists(self.log_path):
            return pd.DataFrame()
        
        df = pd.read_csv(self.log_path)
        return df.tail(n)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics from the log
        """
        if not os.path.exists(self.log_path):
            return {}
        
        df = pd.read_csv(self.log_path)
        
        if df.empty:
            return {}
        
        stats = {
            'total_predictions': len(df),
            'total_fraud_predictions': df['is_fraud'].sum(),
            'fraud_rate': df['is_fraud'].mean(),
            'average_probability': df['probability'].mean(),
            'average_processing_time_ms': df['processing_time_ms'].mean(),
            'total_amount_processed': df['amount'].sum(),
            'average_transaction_amount': df['amount'].mean(),
            'model_versions': df['model_version'].unique().tolist()
        }
        
        return stats
    
    def get_fraud_predictions(self, n: int = 50) -> pd.DataFrame:
        """
        Get recent fraud predictions
        
        Args:
            n: Number of fraud predictions to retrieve
        """
        if not os.path.exists(self.log_path):
            return pd.DataFrame()
        
        df = pd.read_csv(self.log_path)
        fraud_df = df[df['is_fraud'] == True]
        return fraud_df.tail(n)
    
    def get_predictions_by_date(self, date: str) -> pd.DataFrame:
        """
        Get predictions for a specific date
        
        Args:
            date: Date in format 'YYYY-MM-DD'
        """
        if not os.path.exists(self.log_path):
            return pd.DataFrame()
        
        df = pd.read_csv(self.log_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date.astype(str)
        
        return df[df['date'] == date]
    
    def clear_old_logs(self, days: int = 30):
        """
        Clear logs older than specified days
        
        Args:
            days: Number of days to keep logs
        """
        if not os.path.exists(self.log_path):
            return
        
        df = pd.read_csv(self.log_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        cutoff_date = datetime.now() - pd.Timedelta(days=days)
        
        df_filtered = df[df['timestamp'] >= cutoff_date]
        df_filtered.to_csv(self.log_path, index=False)
        
        print(f"Cleared logs older than {days} days")
    
    def export_logs(self, export_path: str, format: str = 'csv'):
        """
        Export logs to a file
        
        Args:
            export_path: Path to export file
            format: Export format ('csv', 'json', 'excel')
        """
        if not os.path.exists(self.log_path):
            print("No logs to export")
            return
        
        df = pd.read_csv(self.log_path)
        
        if format == 'csv':
            df.to_csv(export_path, index=False)
        elif format == 'json':
            df.to_json(export_path, orient='records', indent=2)
        elif format == 'excel':
            df.to_excel(export_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Logs exported to {export_path}")


class AuditLogger:
    """
    Logs audit events for compliance and monitoring
    """
    
    def __init__(self, log_dir='logs', log_file='audit.log'):
        """
        Initialize audit logger
        
        Args:
            log_dir: Directory to store log files
            log_file: Name of the audit log file
        """
        self.log_dir = log_dir
        self.log_file = log_file
        self.log_path = os.path.join(log_dir, log_file)
        
        os.makedirs(log_dir, exist_ok=True)
    
    def log_event(self, event_type: str, details: Dict[str, Any]):
        """
        Log an audit event
        
        Args:
            event_type: Type of event (e.g., 'model_loaded', 'prediction_made', 'error')
            details: Dictionary with event details
        """
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            'event_type': event_type,
            'details': details
        }
        
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_model_loaded(self, model_name: str, model_version: str):
        """Log model loading event"""
        self.log_event('model_loaded', {
            'model_name': model_name,
            'model_version': model_version
        })
    
    def log_prediction_made(self, prediction: int, probability: float, features: Dict):
        """Log prediction event"""
        self.log_event('prediction_made', {
            'prediction': prediction,
            'probability': probability,
            'amount': features.get('Amount', 0)
        })
    
    def log_error(self, error_type: str, error_message: str):
        """Log error event"""
        self.log_event('error', {
            'error_type': error_type,
            'error_message': error_message
        })
    
    def get_audit_trail(self, n: int = 100) -> List[Dict]:
        """
        Get recent audit trail entries
        
        Args:
            n: Number of entries to retrieve
        """
        if not os.path.exists(self.log_path):
            return []
        
        entries = []
        with open(self.log_path, 'r') as f:
            for line in f:
                entries.append(json.loads(line.strip()))
        
        return entries[-n:]


if __name__ == "__main__":
    # Example usage
    print("Logging Module for FraudShield")
    print("This module provides prediction logging and audit trail functionality")
