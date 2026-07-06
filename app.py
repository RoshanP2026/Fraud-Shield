import streamlit as st

st.title("FraudShield 🚀")
st.success("Streamlit deployment working!")

st.write("If you see this, deployment is OK.")

import os
import sys
import joblib
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from PIL import Image

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from CreditCardFraudDetection.utils import (
    get_or_create_dataset,
    plot_amount_distribution,
    plot_time_distribution,
    plot_correlation_heatmap,
    plot_feature_boxplots,
    plot_confusion_matrix_plotly,
    plot_roc_curve_plotly,
    plot_pr_curve_plotly,
    evaluate_classifier
)
from explainability import ModelExplainer
from logger import PredictionLogger, AuditLogger
from feature_engineering import FeatureEngineer

# -------------------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------------------

try:
    logo_path = os.path.join(os.path.dirname(__file__), "favicon.png")
    if os.path.exists(logo_path):
        logo_img = Image.open(logo_path)
    else:
        logo_img = None
except Exception:
    logo_img = None

st.set_page_config(
    page_title="FraudShield - Credit Card Fraud Detection",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
    
    body {
        font-family: 'Inter', sans-serif;
        background-color: #0B0F19;
    }
    
    .stApp {
        background-color: #0B0F19;
    }
    
    .main-title {
        font-family: 'Space Grotesk', sans-serif;
        background: linear-gradient(135deg, #FFFFFF 0%, #38BDF8 50%, #00F2FE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 700;
        letter-spacing: -0.06rem;
        margin-bottom: 0.1rem;
        text-shadow: 0 0 30px rgba(56, 189, 248, 0.15);
    }
    
    .subtitle {
        font-family: 'Inter', sans-serif;
        color: #94A3B8;
        font-size: 1.2rem;
        font-weight: 400;
        margin-bottom: 2.5rem;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #151B2E 0%, #101626 100%);
        border: 1px solid #24314E;
        border-top: 4px solid #38BDF8;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.4);
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        margin-bottom: 1rem;
    }
    
    .metric-card:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 20px 40px rgba(56, 189, 248, 0.25);
        border-color: #00F2FE;
    }
    
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.4rem;
        font-weight: 700;
        color: #FFFFFF;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.1);
    }
    
    .metric-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.85rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .badge-fraud {
        background-color: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 0.4rem 1rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 0.05em;
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.2);
    }
    
    .badge-legit {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 0.4rem 1rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 0.05em;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.2);
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        color: #94A3B8 !important;
        border-bottom: 2px solid transparent !important;
        transition: all 0.2s ease !important;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #38BDF8 !important;
        border-bottom: 2px solid #38BDF8 !important;
    }

    .info-container {
        background-color: #151B2E;
        border-radius: 12px;
        border-left: 5px solid #38BDF8;
        padding: 1.25rem;
        margin: 1rem 0;
    }

    .super-card {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# DATA LOADING & CACHING
# -------------------------------------------------------------------------

@st.cache_data
def load_dataset_cached():
    df = get_or_create_dataset()
    df = df.dropna()
    return df

@st.cache_resource
def load_model_and_scaler():
    """Load trained model and scaler"""
    model_path = "saved_models/random_forest.pkl"
    scaler_path = "saved_models/scaler.pkl"
    
    # Fallback to CreditCardFraudDetection directory
    if not os.path.exists(model_path):
        model_path = "CreditCardFraudDetection/model.pkl"
        scaler_path = "CreditCardFraudDetection/scaler.pkl"
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    else:
        return None, None

def load_ml_assets():
    """Load ML assets with fallback"""
    model, scaler = load_model_and_scaler()
    
    if model is None:
        st.warning("No pre-trained model found. Some features may be limited.")
    
    return model, scaler

# Initialize logger
logger = PredictionLogger()
audit_logger = AuditLogger()

# -------------------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------------------

logo_path = os.path.join(os.path.dirname(__file__), "logo.jpg")
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, use_container_width=True)

st.sidebar.title("FraudShield Control Panel")

page = st.sidebar.radio(
    "Navigate to:",
    ["Home", "Predict", "Analytics", "Model Performance", "Error Analysis", "About"]
)

st.sidebar.markdown("---")

# System Statistics
st.sidebar.subheader("System Statistics")
stats = logger.get_statistics()
if stats:
    st.sidebar.metric("Total Predictions", stats.get('total_predictions', 0))
    st.sidebar.metric("Fraud Rate", f"{stats.get('fraud_rate', 0):.2%}")
    st.sidebar.metric("Avg Probability", f"{stats.get('average_probability', 0):.2%}")

st.sidebar.markdown("---")

st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem; border-radius: 12px; background: linear-gradient(135deg, #151B2E 0%, #101626 100%); border: 1px solid #24314E; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
    <p style="margin: 0; font-size: 0.8rem; color: #94A3B8; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;">Developed by</p>
    <p style="margin: 0.25rem 0; font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.15rem; color: #FFFFFF; text-shadow: 0 0 10px rgba(255,255,255,0.1);">Roshan Perera</p>
    <p style="margin: 0; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; font-weight: 600; color: #38BDF8;">Student ID: S25026203</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------------

df = load_dataset_cached()
model, scaler = load_ml_assets()

# -------------------------------------------------------------------------
# PAGE: HOME
# -------------------------------------------------------------------------

if page == "Home":
    st.markdown('<h1 class="main-title">FraudShield</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced Real-Time Credit Card Fraud Detection with Explainable AI</p>', unsafe_allow_html=True)
    
    banner_path = os.path.join(os.path.dirname(__file__), "banner.jpg")
    if os.path.exists(banner_path):
        st.image(banner_path, use_container_width=True)
    
    st.markdown("""
    ### System Overview
    FraudShield is an enterprise-grade fraud detection system that leverages advanced machine learning techniques 
    to identify fraudulent credit card transactions in real-time.
    
    ### Key Features
    - **Multiple ML Models**: Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, Neural Networks
    - **Explainable AI**: SHAP-based feature attribution for model interpretability
    - **Class Imbalance Handling**: SMOTE, Random Under Sampling, Balanced Random Forest
    - **Hyperparameter Tuning**: GridSearchCV and RandomizedSearchCV optimization
    - **Advanced Feature Engineering**: Transaction velocity, risk scores, interaction features
    - **Real-time Monitoring**: Comprehensive logging and audit trail
    """)
    
    # Dashboard Highlights
    st.markdown("### System Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    
    total_tx = len(df)
    fraud_tx = int(df['Class'].sum())
    fraud_pct = (fraud_tx / total_tx) * 100
    avg_amt = df['Amount'].mean()
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid #38BDF8;">
            <div class="metric-label">Total Transactions</div>
            <div class="metric-value" style="color: #FFFFFF;">{total_tx:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid #F87171;">
            <div class="metric-label">Fraud Cases</div>
            <div class="metric-value" style="color: #F87171;">{fraud_tx}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid #FBBF24;">
            <div class="metric-label">Fraud Proportion</div>
            <div class="metric-value" style="color: #FBBF24;">{fraud_pct:.3f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid #34D399;">
            <div class="metric-label">Average Amount</div>
            <div class="metric-value" style="color: #34D399;">${avg_amt:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Recent Activity
    st.markdown("### Recent Activity")
    recent_predictions = logger.get_recent_predictions(5)
    if not recent_predictions.empty:
        st.dataframe(recent_predictions[['timestamp', 'prediction', 'probability', 'amount']], use_container_width=True)
    else:
        st.info("No recent predictions logged yet.")

# -------------------------------------------------------------------------
# PAGE: PREDICT
# -------------------------------------------------------------------------

elif page == "Predict":
    st.markdown('<h1 class="main-title">Real-Time Transaction Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Evaluate transaction risk with AI-powered fraud detection</p>', unsafe_allow_html=True)
    
    if model is None:
        st.error("No model loaded. Please train a model first using the training script.")
        st.info("Run `python train.py` to train and save models.")
    else:
        # Template Selection
        st.markdown("### Quick Templates")
        template_choice = st.radio(
            "Select a scenario template:",
            [
                "Standard Legitimate Transaction",
                "High-Value Suspicious Purchase",
                "Rapid Micro-Payments",
                "Custom Manual Entry"
            ],
            horizontal=True
        )
        
        # Set defaults based on template
        if template_choice == "Standard Legitimate Transaction":
            val_amount, val_time, val_v14, val_v17, val_v4, val_v11 = 45.20, 45000, 0.50, 0.20, -0.10, 0.00
        elif template_choice == "High-Value Suspicious Purchase":
            val_amount, val_time, val_v14, val_v17, val_v4, val_v11 = 3250.00, 12000, -4.80, -5.20, 3.80, 2.90
        elif template_choice == "Rapid Micro-Payments":
            val_amount, val_time, val_v14, val_v17, val_v4, val_v11 = 8.50, 3600, -3.20, -2.80, 2.50, 3.00
        else:
            val_amount, val_time, val_v14, val_v17, val_v4, val_v11 = 124.50, 86400, 0.10, 0.05, -0.05, 0.02
        
        # Input Form
        st.markdown("### 📝 Transaction Details")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tx_amount = st.number_input("Transaction Amount ($)", min_value=0.01, max_value=100000.0, value=val_amount, step=10.0)
            tx_time = st.number_input("Transaction Time (seconds)", min_value=0, max_value=172800, value=int(val_time), step=3600)
        
        with col2:
            st.markdown("**Key Fraud Indicators**")
            feat_v14 = st.slider("V14 (Negative Correlation)", -15.0, 15.0, val_v14)
            feat_v17 = st.slider("V17 (Negative Correlation)", -15.0, 15.0, val_v17)
        
        with col3:
            st.markdown("**Secondary Indicators**")
            feat_v4 = st.slider("V4 (Positive Correlation)", -15.0, 15.0, val_v4)
            feat_v11 = st.slider("V11 (Positive Correlation)", -15.0, 15.0, feat_v11)
        
        # Prediction Button
        if st.button("🔍 Analyze Transaction", use_container_width=True, type="primary"):
            start_time = time.time()
            
            # Prepare input
            input_data = {f'V{i}': 0.0 for i in range(1, 29)}
            input_data['Time'] = tx_time
            input_data['Amount'] = tx_amount
            input_data['V14'] = feat_v14
            input_data['V17'] = feat_v17
            input_data['V4'] = feat_v4
            input_data['V11'] = feat_v11
            
            ordered_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
            df_input = pd.DataFrame([input_data])[ordered_cols]
            
            # Scale
            if scaler:
                df_input_scaled = df_input.copy()
                df_input_scaled[['Time', 'Amount']] = scaler.transform(df_input[['Time', 'Amount']])
            else:
                df_input_scaled = df_input
            
            # Predict
            prediction = model.predict(df_input_scaled)[0]
            
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(df_input_scaled)[0]
                prob_fraud = probabilities[1]
                prob_legit = probabilities[0]
            else:
                dfunc = model.decision_function(df_input_scaled)[0]
                prob_fraud = 1 / (1 + np.exp(-dfunc))
                prob_legit = 1 - prob_fraud
            
            processing_time = (time.time() - start_time) * 1000
            
            # Log prediction
            logger.log_prediction({
                'prediction': int(prediction),
                'probability': float(prob_fraud),
                'is_fraud': bool(prediction == 1),
                'features': input_data
            }, processing_time_ms=processing_time)
            
            # Display Results
            col_res1, col_res2 = st.columns([1, 1])
            
            with col_res1:
                st.subheader("Prediction Result")
                if prediction == 1 or prob_fraud >= 0.5:
                    st.markdown('<div style="text-align: center; padding: 2rem; border-radius: 0.5rem; background-color: #FEE2E2; border: 1px solid #EF4444;"><span class="badge-fraud" style="font-size: 1.5rem; padding: 0.5rem 1.5rem;">FRAUDULENT TRANSACTION DETECTED</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="text-align: center; padding: 2rem; border-radius: 0.5rem; background-color: #D1FAE5; border: 1px solid #10B981;"><span class="badge-legit" style="font-size: 1.5rem; padding: 0.5rem 1.5rem;">LEGITIMATE TRANSACTION</span></div>', unsafe_allow_html=True)
                
                st.markdown(f"**Legitimate Probability:** `{prob_legit*100:.2f}%`")
                st.markdown(f"**Fraudulent Probability:** `{prob_fraud*100:.2f}%`")
                st.markdown(f"**Confidence:** `{max(prob_fraud, prob_legit)*100:.2f}%`")
                st.markdown(f"**Processing Time:** `{processing_time:.2f}ms`")
            
            with col_res2:
                st.subheader("Risk Gauge")
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob_fraud * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Fraud Risk Index (%)", 'font': {'size': 18}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "#EF4444" if prob_fraud >= 0.5 else "#10B981"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 20], 'color': '#E8F5E9'},
                            {'range': [20, 50], 'color': '#FFF9C4'},
                            {'range': [50, 100], 'color': '#FFEBEE'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Feature Attribution
            st.subheader("Feature Attribution")
            attribution_items = []
            if feat_v14 < -2.0:
                attribution_items.append(f"**V14 Extreme Drop**: {feat_v14:.2f} (strong fraud indicator)")
            if feat_v17 < -2.0:
                attribution_items.append(f"**V17 Extreme Drop**: {feat_v17:.2f} (strong fraud indicator)")
            if feat_v4 > 1.5:
                attribution_items.append(f"**V4 Positive Shift**: {feat_v4:.2f} (elevated fraud risk)")
            if tx_amount > 1000.0:
                attribution_items.append(f"**High Transaction Amount**: ${tx_amount:.2f} (unusual value)")
            
            if attribution_items:
                for item in attribution_items:
                    st.markdown(item)
            else:
                st.success("No significant risk factors detected")

# -------------------------------------------------------------------------
# PAGE: ANALYTICS
# -------------------------------------------------------------------------

elif page == "Analytics":
    st.markdown('<h1 class="main-title">Exploratory Data Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Comprehensive data visualization and insights</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Distribution Analysis", "Correlation Analysis", "Feature Analysis", "Advanced Analytics"])
    
    with tab1:
        st.markdown("### Transaction Amount & Time Distributions")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_amount_distribution(df), use_container_width=True)
        with col2:
            st.plotly_chart(plot_time_distribution(df), use_container_width=True)
        st.info("Fraudulent transactions often exhibit different distribution patterns compared to legitimate ones.")
    
    with tab2:
        st.markdown("### Feature Correlation Heatmap")
        st.plotly_chart(plot_correlation_heatmap(df, num_features=15), use_container_width=True)
        st.markdown("""
        ### Key Insights:
        - V14, V17, V12, V10 show strong negative correlation with fraud
        - V4, V11 show positive correlation with fraud
        - PCA features are generally uncorrelated with each other
        """)
    
    with tab3:
        st.markdown("### Feature Separation Analysis")
        st.plotly_chart(plot_feature_boxplots(df), use_container_width=True)
        st.markdown("""
        ### Feature Separation:
        - V14 and V17 show clear separation between classes
        - These features are highly predictive of fraud
        """)
    
    with tab4:
        st.markdown("### Advanced Analytics")
        st.subheader("Class Distribution")
        class_counts = df['Class'].value_counts()
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Legitimate', 'Fraudulent'],
            values=[class_counts[0], class_counts[1]],
            hole=.4,
            marker_colors=['#34D399', '#F87171']
        )]).
        fig_pie.update_layout(title="Class Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)

# -------------------------------------------------------------------------
# PAGE: MODEL PERFORMANCE
# -------------------------------------------------------------------------

elif page == "Model Performance":
    st.markdown('<h1 class="main-title">Model Performance Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Comprehensive evaluation metrics and model comparison</p>', unsafe_allow_html=True)
    
    if model is None:
        st.error("No model loaded for evaluation.")
    else:
        from sklearn.model_selection import train_test_split
        
        df_clean = df.dropna()
        X = df_clean.drop(columns=['Class'])
        y = df_clean['Class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
        
        X_test_scaled = X_test.copy()
        if scaler:
            X_test_scaled[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])
        
        y_pred = model.predict(X_test_scaled)
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            dfunc = model.decision_function(X_test_scaled)
            y_prob = 1 / (1 + np.exp(-dfunc))
        
        metrics = evaluate_classifier(y_test, y_pred, y_prob)
        
        # Metrics Display
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.4f}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.4f}")
        with col4:
            st.metric("F1-Score", f"{metrics['f1']:.4f}")
        
        # Confusion Matrix
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Confusion Matrix")
            st.plotly_chart(plot_confusion_matrix_plotly(metrics['confusion_matrix']), use_container_width=True)
        
        with col2:
            st.subheader("Classification Report")
            rep_df = pd.DataFrame(metrics['class_report']).transpose()
            st.dataframe(rep_df.style.format("{:.4f}"), use_container_width=True)
        
        # ROC and PR Curves
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("ROC Curve")
            st.plotly_chart(plot_roc_curve_plotly(y_test, y_prob, type(model).__name__), use_container_width=True)
        
        with col2:
            st.subheader("Precision-Recall Curve")
            st.plotly_chart(plot_pr_curve_plotly(y_test, y_prob, type(model).__name__), use_container_width=True)

# -------------------------------------------------------------------------
# PAGE: ERROR ANALYSIS
# -------------------------------------------------------------------------

elif page == "Error Analysis":
    st.markdown('<h1 class="main-title">Error Analysis & Diagnostics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Deep dive into model errors and misclassifications</p>', unsafe_allow_html=True)
    
    if model is None:
        st.error("No model loaded for error analysis.")
    else:
        from sklearn.model_selection import train_test_split
        
        df_clean = df.dropna()
        X = df_clean.drop(columns=['Class'])
        y = df_clean['Class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
        
        X_test_scaled = X_test.copy()
        if scaler:
            X_test_scaled[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])
        
        y_pred = model.predict(X_test_scaled)
        
        # Identify errors
        false_positives = X_test[(y_test == 0) & (y_pred == 1)]
        false_negatives = X_test[(y_test == 1) & (y_pred == 0)]
        
        st.markdown("### Error Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("False Positives", len(false_positives))
        with col2:
            st.metric("False Negatives", len(false_negatives))
        with col3:
            st.metric("Total Errors", len(false_positives) + len(false_negatives))
        
        # False Positive Analysis
        st.markdown("---")
        st.subheader("False Positive Analysis")
        st.info("Legitimate transactions incorrectly flagged as fraud")
        
        if len(false_positives) > 0:
            fp_sample = false_positives.head(5)
            st.dataframe(fp_sample[['Time', 'Amount', 'V14', 'V17', 'V4', 'V11']], use_container_width=True)
            
            st.markdown("**Common Characteristics:**")
            fp_avg_amount = false_positives['Amount'].mean()
            fp_avg_v14 = false_positives['V14'].mean()
            st.markdown(f"- Average Amount: ${fp_avg_amount:.2f}")
            st.markdown(f"- Average V14: {fp_avg_v14:.2f}")
        else:
            st.success("No false positives detected!")
        
        # False Negative Analysis
        st.markdown("---")
        st.subheader("False Negative Analysis")
        st.warning("Fraudulent transactions missed by the model")
        
        if len(false_negatives) > 0:
            fn_sample = false_negatives.head(5)
            st.dataframe(fn_sample[['Time', 'Amount', 'V14', 'V17', 'V4', 'V11']], use_container_width=True)
            
            st.markdown("**Common Characteristics:**")
            fn_avg_amount = false_negatives['Amount'].mean()
            fn_avg_v14 = false_negatives['V14'].mean()
            st.markdown(f"- Average Amount: ${fn_avg_amount:.2f}")
            st.markdown(f"- Average V14: {fn_avg_v14:.2f}")
        else:
            st.success("No false negatives detected!")
        
        # Recommendations
        st.markdown("---")
        st.subheader("Recommendations")
        st.markdown("""
        - **For False Positives**: Consider adjusting the decision threshold to reduce false alarms
        - **For False Negatives**: Increase model complexity or add more training data for fraud cases
        - **Feature Engineering**: Create additional features that better separate edge cases
        - **Ensemble Methods**: Combine multiple models to improve overall performance
        """)

# -------------------------------------------------------------------------
# PAGE: ABOUT & DIAGNOSTICS
# -------------------------------------------------------------------------

elif page == "About":
    st.markdown('<h1 class="main-title">About</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #F8FAFC; padding: 2rem; border-radius: 12px; border: 1px solid #E2E8F0; margin: 2rem 0;">
        <h2 style="margin: 0 0 1rem 0; font-size: 1.8rem; font-weight: 700; color: #0F172A;">Roshan Perera</h2>
        <p style="margin: 0.5rem 0; font-family: 'JetBrains Mono', monospace; font-size: 1rem; font-weight: 600; color: #0EA5E9;">Student ID: S25026203</p>
        <p style="margin: 1rem 0 0 0; font-size: 1rem; color: #475569; line-height: 1.6;">MSc Advanced Machine Learning & Financial Risk Mitigation</p>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------------------

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem 1rem; color: #64748B; font-size: 0.9rem; border-top: 1px solid #24314E; margin-top: 2rem;">
    <p style="margin: 0 0 0.5rem 0; font-weight: 600; color: #94A3B8;">FraudShield Credit Card Fraud Detection System</p>
    <p style="margin: 0.25rem 0; font-size: 0.85rem;">MSc Advanced Machine Learning Assignment</p>
    <p style="margin: 0.25rem 0; font-size: 0.85rem;">Student ID: S25026203</p>
    <p style="margin: 1rem 0 0 0; font-size: 0.8rem; color: #475569;">© 2026 Roshan Perera</p>
</div>
""", unsafe_allow_html=True)
