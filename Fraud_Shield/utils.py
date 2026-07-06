"""
Credit Card Fraud Detection System - Utilities Module
Author: Senior ML Engineer & Python Developer
MSc Advanced Machine Learning Assignment Reference
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import learning_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

# -------------------------------------------------------------------------
# 0. THEME CONFIGURATION (Super App Premium Dark Styling)
# -------------------------------------------------------------------------

def apply_premium_dark_style(fig):
    """
    Applies unified premium dark-theme branding to any Plotly figure.
    """
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#151B2E',
        plot_bgcolor='#101626',
        font=dict(color='#F1F5F9', family='Inter, Space Grotesk, sans-serif'),
        xaxis=dict(
            gridcolor='#1E293B',
            zerolinecolor='#1E293B',
            linecolor='#24314E'
        ),
        yaxis=dict(
            gridcolor='#1E293B',
            zerolinecolor='#1E293B',
            linecolor='#24314E'
        )
    )
    return fig

# -------------------------------------------------------------------------
# 1. DATA GENERATION & SEEDING (Kaggle Credit Card Fraud Dataset Schema)
# -------------------------------------------------------------------------

def generate_credit_card_data(num_samples=50000, fraud_rate=0.005, random_state=42):
    """
    Generates a highly realistic synthetic credit card fraud dataset mimicking 
    the PCA-transformed Kaggle Credit Card Fraud dataset schema.
    
    Kaggle features:
    - Time: Seconds elapsed since the first transaction.
    - V1 to V28: PCA features. In the real dataset, certain features like V17, V14, V12, V10 
                are strongly negatively correlated with fraud, while V4 and V11 are positively correlated.
    - Amount: Transaction amount.
    - Class: 1 for Fraud, 0 for Legitimate.
    """
    np.random.seed(random_state)
    
    num_fraud = max(int(num_samples * fraud_rate), 50)  # Ensure at least 50 fraud cases
    num_legit = num_samples - num_fraud
    
    # Generate Legitimate Transactions (Class 0)
    legit_data = {}
    legit_data['Time'] = np.random.uniform(0, 172800, num_legit) # 2 days of transactions
    
    # V1 to V28 for legit (mostly standard normal, with slight variation)
    for i in range(1, 29):
        legit_data[f'V{i}'] = np.random.normal(0, 1.0, num_legit)
        
    # Amount (Log-Normal distribution for transaction amounts)
    legit_data['Amount'] = np.random.lognormal(mean=2.8, sigma=1.0, size=num_legit)
    legit_data['Class'] = np.zeros(num_legit, dtype=int)
    
    df_legit = pd.DataFrame(legit_data)
    
    # Generate Fraud Transactions (Class 1)
    fraud_data = {}
    fraud_data['Time'] = np.random.uniform(0, 172800, num_fraud)
    
    # V1 to V28 for fraud (highly shifted distributions for key PCA variables)
    # This reflects the distinct distributions observed in the real Kaggle dataset.
    for i in range(1, 29):
        if i in [17, 14, 12, 10]:  # Strongly negative correlated features
            fraud_data[f'V{i}'] = np.random.normal(-4.5, 2.0, num_fraud)
        elif i in [4, 11]:         # Strongly positive correlated features
            fraud_data[f'V{i}'] = np.random.normal(3.5, 1.5, num_fraud)
        elif i in [1, 3, 7, 16]:   # Moderately negative correlated features
            fraud_data[f'V{i}'] = np.random.normal(-2.0, 1.5, num_fraud)
        else:                      # Uncorrelated features
            fraud_data[f'V{i}'] = np.random.normal(0.0, 1.2, num_fraud)
            
    # Amount for fraud (often higher on average, with larger variance)
    fraud_data['Amount'] = np.random.lognormal(mean=4.2, sigma=1.2, size=num_fraud)
    fraud_data['Class'] = np.ones(num_fraud, dtype=int)
    
    df_fraud = pd.DataFrame(fraud_data)
    
    # Combine and Shuffle
    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Clean up outlier values or negative amounts
    df['Amount'] = df['Amount'].round(2)
    df['Time'] = df['Time'].astype(int)
    
    # Ensure no NaN values
    df = df.dropna()
    
    return df

def get_or_create_dataset(filepath="CreditCardFraudDetection/creditcard.csv", num_samples=50000, fraud_rate=0.005):
    """
    Loads dataset if exists, otherwise generates and saves it to CSV.
    """
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df = generate_credit_card_data(num_samples=num_samples, fraud_rate=fraud_rate)
        df.to_csv(filepath, index=False)
        return df

# -------------------------------------------------------------------------
# 2. EDA PLOTTING FUNCTIONS
# -------------------------------------------------------------------------

def plot_amount_distribution(df):
    """
    Returns a Plotly Figure showing the distribution of transaction amounts for Class 0 vs Class 1.
    """
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df[df['Class'] == 0]['Amount'],
        name='Legitimate',
        xbins=dict(start=0, end=1000, size=20),
        marker_color='#34D399',
        opacity=0.7,
        histnorm='probability density'
    ))
    fig.add_trace(go.Histogram(
        x=df[df['Class'] == 1]['Amount'],
        name='Fraudulent',
        xbins=dict(start=0, end=1000, size=20),
        marker_color='#F87171',
        opacity=0.7,
        histnorm='probability density'
    ))
    
    fig.update_layout(
        title_text='Transaction Amount Probability Density (Truncated at $1000)',
        barmode='overlay',
        xaxis_title_text='Amount ($)',
        yaxis_title_text='Density',
        legend=dict(x=0.8, y=0.9, bgcolor='rgba(21, 27, 46, 0.8)'),
        height=400,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return apply_premium_dark_style(fig)

def plot_time_distribution(df):
    """
    Returns a Plotly Figure showing transaction volume over Time (seconds).
    """
    fig = go.Figure()
    # Legitimate (Time in hours)
    fig.add_trace(go.Histogram(
        x=df[df['Class'] == 0]['Time'] / 3600,
        name='Legitimate',
        nbinsx=48,
        marker_color='#38BDF8',
        opacity=0.7,
        histnorm='probability density'
    ))
    # Fraudulent
    fig.add_trace(go.Histogram(
        x=df[df['Class'] == 1]['Time'] / 3600,
        name='Fraudulent',
        nbinsx=48,
        marker_color='#FBBF24',
        opacity=0.7,
        histnorm='probability density'
    ))
    
    fig.update_layout(
        title_text='Transaction Distribution Over Time (Hours)',
        barmode='overlay',
        xaxis_title_text='Time (Hours from first transaction)',
        yaxis_title_text='Density',
        legend=dict(x=0.8, y=0.9, bgcolor='rgba(21, 27, 46, 0.8)'),
        height=400,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return apply_premium_dark_style(fig)

def plot_correlation_heatmap(df, num_features=15):
    """
    Returns a Plotly Heatmap showing the correlation matrix for the top features.
    """
    # Calculate correlations with class to find top correlated features
    correlations = df.corr()['Class'].abs().sort_values(ascending=False)
    top_cols = correlations.index[:num_features].tolist()
    
    corr_matrix = df[top_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale='RdBu_r',
        labels=dict(color="Correlation"),
        title=f"Correlation Matrix (Top {num_features} Features correlated with Class)"
    )
    fig.update_layout(
        height=500,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return apply_premium_dark_style(fig)

def plot_feature_boxplots(df, features_to_plot=['V14', 'V17', 'V4', 'V11']):
    """
    Returns a Plotly Boxplot matrix comparing distributions for Class 0 vs Class 1.
    """
    fig = go.Figure()
    
    for feat in features_to_plot:
        fig.add_trace(go.Box(
            y=df[df['Class'] == 0][feat],
            name=f'{feat} (Legit)',
            marker_color='#34D399',
            boxpoints='outliers'
        ))
        fig.add_trace(go.Box(
            y=df[df['Class'] == 1][feat],
            name=f'{feat} (Fraud)',
            marker_color='#F87171',
            boxpoints='outliers'
        ))
        
    fig.update_layout(
        title='Distribution Contrast of Distinct Features (Class 0 vs Class 1)',
        yaxis_title_text='Feature Values',
        height=450,
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return apply_premium_dark_style(fig)

# -------------------------------------------------------------------------
# 3. MODEL EVALUATION METRICS & PLOTS (MSc Level)
# -------------------------------------------------------------------------

def evaluate_classifier(y_true, y_pred, y_prob):
    """
    Returns a dictionary of standard and advanced ML evaluation metrics.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'confusion_matrix': cm,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'class_report': classification_report(y_true, y_pred, output_dict=True)
    }
    return metrics

def plot_confusion_matrix_plotly(cm):
    """
    Returns a Plotly Confusion Matrix Heatmap.
    """
    z = cm
    x = ['Predicted Legitimate', 'Predicted Fraudulent']
    y = ['Actual Legitimate', 'Actual Fraudulent']
    
    # Annot text
    txt = [[str(val) for val in row] for row in z]
    
    fig = px.imshow(
        z, x=x, y=y,
        text_auto=True,
        color_continuous_scale='Blues',
        title="Confusion Matrix Visualization"
    )
    fig.update_layout(
        height=350,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return apply_premium_dark_style(fig)

def plot_roc_curve_plotly(y_true, y_prob, model_name="Model"):
    """
    Returns a Plotly ROC Curve.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    
    fig = go.Figure()
    # Diagonal baseline
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='#475569'), name='Random Guess'))
    # ROC Curve
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', line=dict(color='#38BDF8', width=3), name=f'{model_name} (AUC = {auc_score:.4f})'))
    
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=400,
        legend=dict(x=0.5, y=0.1, bgcolor='rgba(21, 27, 46, 0.8)'),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return apply_premium_dark_style(fig)

def plot_pr_curve_plotly(y_true, y_prob, model_name="Model"):
    """
    Returns a Plotly Precision-Recall Curve.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', line=dict(color='#F43F5E', width=3), name=f'{model_name} PR Curve'))
    
    fig.update_layout(
        title='Precision-Recall (PR) Curve (Optimized for Imbalanced Data)',
        xaxis_title='Recall',
        yaxis_title='Precision',
        height=400,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return apply_premium_dark_style(fig)

def plot_learning_curve_plt(estimator, X, y, cv=3, scoring='f1'):
    """
    Generates learning curves using Matplotlib and returns the Figure.
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5), random_state=42
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='#151B2E')
    ax.set_facecolor('#101626')
    ax.grid(True, linestyle='--', color='#1E293B', alpha=0.5)
    
    # Text and grid styles matching premium dark mode
    ax.tick_params(colors='#94A3B8')
    ax.xaxis.label.set_color('#94A3B8')
    ax.yaxis.label.set_color('#94A3B8')
    ax.title.set_color('#F1F5F9')
    
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#F87171")
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color="#34D399")
    
    ax.plot(train_sizes, train_mean, 'o-', color="#F87171", linewidth=2.5, label="Training Score")
    ax.plot(train_sizes, test_mean, 'o-', color="#34D399", linewidth=2.5, label="Cross-Validation Score")
    
    ax.set_title(f"Learning Curve ({scoring.upper()} Score)")
    ax.set_xlabel("Training Examples")
    ax.set_ylabel(scoring.upper())
    
    legend = ax.legend(loc="best", facecolor='#151B2E', edgecolor='#24314E')
    plt.setp(legend.get_texts(), color='#F1F5F9')
    
    for spine in ax.spines.values():
        spine.set_color('#24314E')
        
    plt.tight_layout()
    return fig

def get_overfitting_explanation():
    """
    Returns a rigorous MSc-level explanation of overfitting in the context of credit card fraud.
    """
    explanation = """
    ### Understanding Overfitting in Imbalanced Credit Card Classification
    
    Overfitting occurs when a machine learning estimator captures noise and idiosyncratic variations in the training sample instead of generalized decision boundaries. 
    
    In the context of highly imbalanced datasets like Credit Card Fraud detection, overfitting manifests through unique mechanisms:
    
    1. **Minority Class Over-Representation (SMOTE Artifacts)**: 
       When applying synthetic oversampling (like SMOTE), generating synthetic samples along line segments connecting existing minority class samples can cause classifiers (especially Random Forests or deep Decision Trees) to create highly complex, circular, or "spaghetti-like" decision boundaries. The model achieves 100% precision and recall on the oversampled training set, but generalizes poorly, flagging innocent, normal transactions in the real validation partition.
       
    2. **Disproportionate Training/Validation Metric Gap**:
       - An overfit Random Forest might yield **Recall: 1.00** and **Precision: 1.00** on training sets, but plunge to **Precision: 0.65** on validation partitions.
       - If training metrics are perfect while validation metrics deteriorate significantly, regularization parameters must be tightened (e.g., restricting `max_depth` in Tree-based models or raising `C` penalty parameters / using L1/L2 lasso-ridge regularization in Logistic Regression).
       
    3. **Generalization Diagnostics**:
       - **Learning Curves**: A large gap between the training score curve and the cross-validation score curve denotes high variance (overfitting). As the number of training examples increases, these curves should ideally converge.
       - **Oversampling leaking**: It is vital that **SMOTE is ONLY performed on the Training fold** and never on the validation/test folds. Applying SMOTE to the entire dataset before splitting causes information leakage, leading to artificially inflated (and highly deceptive) performance metrics.
    """
    return explanation
