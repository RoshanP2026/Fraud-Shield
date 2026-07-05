import os
import sys
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.append(os.path.dirname(__file__))

from utils import (
    get_or_create_dataset,
    plot_amount_distribution,
    plot_time_distribution,
    plot_correlation_heatmap,
    plot_feature_boxplots,
    plot_confusion_matrix_plotly,
    plot_roc_curve_plotly,
    plot_pr_curve_plotly,
    get_overfitting_explanation,
    evaluate_classifier
)

from PIL import Image

try:
    logo_path = os.path.join(os.path.dirname(__file__), "favicon.png")
    if os.path.exists(logo_path):
        logo_img = Image.open(logo_path)
    else:
        logo_img = "💳"
except Exception:
    logo_img = "💳"

st.set_page_config(
    page_title="FraudShield - Advanced Credit Card Fraud Detection System",
    page_icon=logo_img,
    layout="wide",
    initial_sidebar_state="expanded"
)

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

@st.cache_data
def load_dataset_cached():
    df = get_or_create_dataset()
    # Clean any NaN values that might exist
    df = df.dropna()
    return df

def train_and_save_fallback_model(df):
    """
    Trains a fast Logistic Regression model as a fallback if no model exists.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    df_clean = df.dropna()
    
    X = df_clean.drop(columns=['Class'])
    y = df_clean['Class']
    
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])
    
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(X_scaled, y)
    
    os.makedirs("CreditCardFraudDetection", exist_ok=True)
    joblib.dump(model, "CreditCardFraudDetection/model.pkl")
    joblib.dump(scaler, "CreditCardFraudDetection/scaler.pkl")
    return model, scaler

def load_ml_assets():
    """
    Loads saved model and scaler, automatically training fallback if missing.
    """
    model_path = "CreditCardFraudDetection/model.pkl"
    scaler_path = "CreditCardFraudDetection/scaler.pkl"
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            return model, scaler
        except Exception as e:
            st.sidebar.error(f"⚠️ Error loading model: {e}")
            with st.sidebar.spinner("Training fallback model..."):
                df = load_dataset_cached()
                return train_and_save_fallback_model(df)
    else:
        st.sidebar.info("🔄 No pre-trained model found. Training initial model...")
        with st.sidebar.spinner("Training initial model..."):
            df = load_dataset_cached()
            return train_and_save_fallback_model(df)

logo_path = os.path.join(os.path.dirname(__file__), "logo.jpg")
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, use_container_width=True)
st.sidebar.title("FraudShield Control Panel")
page = st.sidebar.radio(
    "Go To Page:",
    ["Home", "Dataset Explorer", "Exploratory Data Analysis", "Model Training", "Prediction System", "Performance Analytics", "About & Diagnostics"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem; border-radius: 12px; background: linear-gradient(135deg, #151B2E 0%, #101626 100%); border: 1px solid #24314E; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
    <p style="margin: 0; font-size: 0.8rem; color: #94A3B8; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;">Developed by</p>
    <p style="margin: 0.25rem 0; font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.15rem; color: #FFFFFF; text-shadow: 0 0 10px rgba(255,255,255,0.1);">Roshan Perera</p>
    <p style="margin: 0; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; font-weight: 600; color: #38BDF8;">ID: S25026203</p>
</div>
""", unsafe_allow_html=True)

df = load_dataset_cached()
model, scaler = load_ml_assets()

if page == "Home":
    col_header_logo, col_header_title = st.columns([1, 6])
    with col_header_logo:
        logo_path = os.path.join(os.path.dirname(__file__), "logo.jpg")
        if os.path.exists(logo_path):
            st.image(logo_path, use_container_width=True)
    with col_header_title:
        st.markdown('<h1 class="main-title">FraudShield</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Securing transactions with advanced real-time Machine Learning intelligence</p>', unsafe_allow_html=True)
    
    banner_path = os.path.join(os.path.dirname(__file__), "banner.jpg")
    if os.path.exists(banner_path):
        st.image(banner_path, use_container_width=True)
    
    st.markdown("""
    ### About
    This system detects credit card fraud using machine learning models.
    Credit card fraud detection is characterized by severe class imbalance (typically < 0.2% of transactions are fraudulent).
    Standard classification objectives (like Accuracy) fail in this regime.
    
    This platform addresses this by:
    1. **SMOTE**: Synthetic Minority Over-sampling Technique to balance training data
    2. **Models**: Logistic Regression, Random Forest, and XGBoost/Gradient Boosting
    3. **Metrics**: Focus on Recall, Precision, and F1-Score
    """)
    
    st.markdown("### System Dashboard Highlights")
    col1, col2, col3, col4 = st.columns(4)
    
    total_tx = len(df)
    fraud_tx = int(df['Class'].sum())
    fraud_pct = (fraud_tx / total_tx) * 100
    avg_amt = df['Amount'].mean()
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid #38BDF8;">
            <div class="metric-label">📊 Total Transactions</div>
            <div class="metric-value" style="color: #FFFFFF;">{total_tx:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid #F87171;">
            <div class="metric-label">🚨 Fraud Cases</div>
            <div class="metric-value" style="color: #F87171;">{fraud_tx}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid #FBBF24;">
            <div class="metric-label">📈 Fraud Proportion</div>
            <div class="metric-value" style="color: #FBBF24;">{fraud_pct:.3f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid #34D399;">
            <div class="metric-label">💰 Average Amount</div>
            <div class="metric-value" style="color: #34D399;">${avg_amt:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

elif page == "Dataset Explorer":
    st.markdown('<h1 class="main-title">Kaggle Credit Card Dataset Explorer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Inspecting raw transactional shapes, PCA structures, and distributions</p>', unsafe_allow_html=True)
    
    st.markdown("""
    The dataset contains transactions made by credit cards in September 2013 by European cardholders. 
    Due to confidentiality issues, the original features have been transformed using PCA into features **V1 to V28**. 
    Only **Time** and **Amount** are untransformed numerical inputs.
    """)
    
    st.subheader("1. Interactive Data Table View")
    row_count = st.slider("Select number of rows to display:", 5, 100, 10)
    st.dataframe(df.head(row_count), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("2. Dataset Shape & Characteristics")
        st.write(f"- **Total Rows (Transactions):** {df.shape[0]:,}")
        st.write(f"- **Total Columns (Features):** {df.shape[1]}")
        st.write(f"- **PCA Columns:** V1 - V28")
        st.write(f"- **Target Variable:** `Class` (0 = Legitimate, 1 = Fraudulent)")
        
        null_count = df.isnull().sum().sum()
        st.write(f"- **Missing/Null Values Count:** {null_count}")
        if null_count == 0:
            st.success("No missing values present.")
            
    with col2:
        st.subheader("3. Class Imbalance Breakdown")
        counts = df['Class'].value_counts()
        pcts = df['Class'].value_counts(normalize=True) * 100
        
        st.write(f"- **Legitimate Transactions (Class 0):** {counts[0]:,} ({pcts[0]:.4f}%)")
        st.write(f"- **Fraudulent Transactions (Class 1):** {counts[1]:,} ({pcts[1]:.4f}%)")
        st.write("- **Imbalance Ratio:** 1 : " + f"{int(counts[0]/counts[1])}")
        
        st.progress(float(pcts[1] / 100) * 10)
        st.caption("Note: The green bar represents the fraud ratio (scaled by 10x for visibility).")

    st.subheader("4. Statistical Summary")
    st.write(df.describe())

elif page == "Exploratory Data Analysis":
    st.markdown('<h1 class="main-title">Interactive Exploratory Data Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Diagnosing variance, correlations, and feature separation boundaries</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Amount & Time Distribution", "Correlation Matrix", "Feature Separation (Boxplots)"])
    
    with tab1:
        st.markdown("### Transaction Amount & Time Profiles")
        st.plotly_chart(plot_amount_distribution(df), use_container_width=True)
        st.plotly_chart(plot_time_distribution(df), use_container_width=True)
        st.info("💡 Fraudulent transactions often exhibit different volume distribution over time compared to normal operations.")
        
    with tab2:
        st.markdown("### Feature Correlation Analysis")
        st.plotly_chart(plot_correlation_heatmap(df, num_features=15), use_container_width=True)
        st.markdown("""
        ### Correlation Analysis
        - V1-V28 features are PCA principal components and are mutually uncorrelated
        - Features like V17, V14, V12, V10 show negative correlation with fraud
        - Features like V4, V11 show positive correlation with fraud
        """)
        
    with tab3:
        st.markdown("### Class Separation via High-Correlation PCA Features")
        st.plotly_chart(plot_feature_boxplots(df), use_container_width=True)
        st.markdown("""
        ### Feature Separation
        The boxplots show divergence in distributions for fraudulent vs legitimate transactions.
        V14 and V17 are shifted downwards for fraud.
        """)

elif page == "Model Training":
    st.markdown('<h1 class="main-title">Machine Learning Training Panel</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Customize oversampling parameters, trigger models, and optimize thresholds</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Training Parameters")
        
        use_smote = st.checkbox("Enable SMOTE Oversampling", value=True)
        smote_ratio = st.slider("SMOTE Minority Ratio (Synthetic/Majority):", 0.05, 0.50, 0.10, 0.05) if use_smote else 0.0
        test_split = st.slider("Validation/Test Split Ratio:", 0.10, 0.30, 0.20, 0.05)
        
        st.markdown("---")
        st.markdown("### Choose Models to Train:")
        train_lr = st.checkbox("Logistic Regression (L2)", value=True)
        train_rf = st.checkbox("Random Forest Classifier", value=True)
        train_xgb = st.checkbox("XGBoost / Gradient Boosting", value=True)
        
        trigger = st.button("🚀 Train and Optimize Models", use_container_width=True)
        
    with col2:
        st.subheader("Live Training Console & Outputs")
        if trigger:
            with st.spinner("Executing Data Preprocessing, Stratified Split, Standard Scaling, and Resampling..."):
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler
                from sklearn.linear_model import LogisticRegression
                from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
                
                has_smote_lib = True
                try:
                    from imblearn.over_sampling import SMOTE
                except ImportError:
                    has_smote_lib = False
                
                data_clean = df.drop_duplicates().dropna().reset_index(drop=True)
                
                X = data_clean.drop(columns=['Class'])
                y = data_clean['Class']
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_split, random_state=42, stratify=y
                )
                
                scaler_fit = StandardScaler()
                X_train_scaled = X_train.copy()
                X_test_scaled = X_test.copy()
                cols_to_scale = ['Time', 'Amount']
                X_train_scaled[cols_to_scale] = scaler_fit.fit_transform(X_train[cols_to_scale])
                X_test_scaled[cols_to_scale] = scaler_fit.transform(X_test[cols_to_scale])
                
                if use_smote and has_smote_lib:
                    sm = SMOTE(sampling_strategy=smote_ratio, random_state=42)
                    X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)
                    st.write(f"✔️ **SMOTE Applied**: Training size from {len(X_train_scaled)} to {len(X_train_res)}.")
                else:
                    X_train_res, y_train_res = X_train_scaled, y_train
                    st.write("ℹ️ Training on original imbalanced data.")
                
                trained_models = {}
                
                if train_lr:
                    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
                    lr.fit(X_train_res, y_train_res)
                    trained_models['Logistic Regression'] = lr
                    st.write("✔️ Trained Logistic Regression (Balanced weights).")
                    
                if train_rf:
                    rf = RandomForestClassifier(n_estimators=30, max_depth=12, class_weight='balanced_subsample', random_state=42, n_jobs=-1)
                    rf.fit(X_train_res, y_train_res)
                    trained_models['Random Forest'] = rf
                    st.write("✔️ Trained Random Forest Classifier (n_estimators=30, max_depth=12).")
                    
                if train_xgb:
                    try:
                        import xgboost as xgb
                        xgb_cl = xgb.XGBClassifier(n_estimators=30, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss')
                        xgb_cl.fit(X_train_res, y_train_res)
                        trained_models['XGBoost'] = xgb_cl
                        st.write("✔️ Trained XGBoost Classifier.")
                    except ImportError:
                        gb = GradientBoostingClassifier(n_estimators=30, max_depth=5, learning_rate=0.1, random_state=42)
                        gb.fit(X_train_res, y_train_res)
                        trained_models['Gradient Boosting (XGB Fallback)'] = gb
                        st.write("✔️ Trained Gradient Boosting Classifier (XGBoost fallback).")
                
                if not trained_models:
                    st.warning("Please check at least one model to train.")
                else:
                    st.write("### Test Set Performance Evaluation")
                    
                    summary_data = []
                    best_f1 = 0
                    best_model_obj = None
                    best_model_name = ""
                    
                    for name, clf in trained_models.items():
                        y_test_pred = clf.predict(X_test_scaled)
                        if hasattr(clf, "predict_proba"):
                            y_test_prob = clf.predict_proba(X_test_scaled)[:, 1]
                        else:
                            y_test_prob = clf.decision_function(X_test_scaled)
                        
                        m = evaluate_classifier(y_test, y_test_pred, y_test_prob)
                        summary_data.append({
                            "Classifier": name,
                            "Accuracy": f"{m['accuracy']:.4f}",
                            "Precision": f"{m['precision']:.4f}",
                            "Recall": f"{m['recall']:.4f}",
                            "F1-Score": f"{m['f1']:.4f}",
                            "ROC-AUC": f"{m['roc_auc']:.4f}"
                        })
                        
                        if m['f1'] > best_f1:
                            best_f1 = m['f1']
                            best_model_obj = clf
                            best_model_name = name
                            
                    summary_df = pd.DataFrame(summary_data)
                    st.table(summary_df)
                    
                    st.success(f"🏆 Best model is **{best_model_name}** with Test F1 Score of **{best_f1:.4f}**")
                    
                    os.makedirs("CreditCardFraudDetection", exist_ok=True)
                    joblib.dump(best_model_obj, "CreditCardFraudDetection/model.pkl")
                    joblib.dump(scaler_fit, "CreditCardFraudDetection/scaler.pkl")
                    st.write("💾 Saved model and scaler.")
                    st.write("Refresh or navigate to pages to load and explore.")
        else:
            st.info("Adjust parameters and click **Train and Optimize Models** to start.")

elif page == "Prediction System":
    st.markdown('<h1 class="main-title">Real-Time Transactional Inference Sandbox</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Select scenario templates or configure manual inputs to evaluate real-time fraud probability scores and predictive attributions</p>', unsafe_allow_html=True)
    
    st.markdown("### 💡 Interactive Scenario Templates")
    template_choice = st.radio(
        "Choose a pre-defined scenario to auto-fill input parameters:",
        [
            "Standard Legitimate Transaction", 
            "High-Value Suspicious Purchase (Anomalous PCA)", 
            "Rapid Micro-Payments (Card-Testing Signature)", 
            "Custom Manual Entry"
        ],
        horizontal=True
    )
    
    # Establish dynamic defaults based on template
    if template_choice == "Standard Legitimate Transaction":
        val_amount = 45.20
        val_time = 45000
        val_v14 = 0.50
        val_v17 = 0.20
        val_v4 = -0.10
        val_v11 = 0.00
    elif template_choice == "High-Value Suspicious Purchase (Anomalous PCA)":
        val_amount = 3250.00
        val_time = 12000
        val_v14 = -4.80
        val_v17 = -5.20
        val_v4 = 3.80
        val_v11 = 2.90
    elif template_choice == "Rapid Micro-Payments (Card-Testing Signature)":
        val_amount = 8.50
        val_time = 3600
        val_v14 = -3.20
        val_v17 = -2.80
        val_v4 = 2.50
        val_v11 = 3.00
    else: # Custom Manual Entry
        val_amount = 124.50
        val_time = 86400
        val_v14 = 0.10
        val_v17 = 0.05
        val_v4 = -0.05
        val_v11 = 0.02

    st.markdown("### Transaction Specifications")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        tx_amount = st.number_input("Transaction Amount ($):", min_value=0.01, max_value=100000.0, value=val_amount, step=10.0)
        tx_time = st.number_input("Transaction Time (Seconds from Day Start):", min_value=0, max_value=172800, value=int(val_time), step=3600)
        
    with col2:
        st.markdown("**Key PCA Features (Highly Correlated with Fraud)**")
        feat_v14 = st.slider("V14 (Negatively Correlated):", -15.0, 15.0, val_v14)
        feat_v17 = st.slider("V17 (Negatively Correlated):", -15.0, 15.0, val_v17)
        
    with col3:
        st.markdown("**Other Influential PCA Features**")
        feat_v4 = st.slider("V4 (Positively Correlated):", -15.0, 15.0, val_v4)
        feat_v11 = st.slider("V11 (Positively Correlated):", -15.0, 15.0, val_v11)
        
    st.markdown("---")
    
    input_cols = [f'V{i}' for i in range(1, 29)]
    input_data = {f'V{i}': 0.0 for i in range(1, 29)}
    
    input_data['Time'] = tx_time
    input_data['Amount'] = tx_amount
    input_data['V14'] = feat_v14
    input_data['V17'] = feat_v17
    input_data['V4'] = feat_v4
    input_data['V11'] = feat_v11
    
    ordered_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    df_input = pd.DataFrame([input_data])[ordered_cols]
    
    if st.button("🔍 Evaluate Transaction Risk Profile", use_container_width=True):
        df_input_scaled = df_input.copy()
        df_input_scaled[['Time', 'Amount']] = scaler.transform(df_input[['Time', 'Amount']])
        
        prediction = model.predict(df_input_scaled)[0]
        
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(df_input_scaled)[0]
            prob_fraud = probabilities[1]
            prob_legit = probabilities[0]
        else:
            dfunc = model.decision_function(df_input_scaled)[0]
            prob_fraud = 1 / (1 + np.exp(-dfunc)) # sigmoid
            prob_legit = 1 - prob_fraud
            
        col_res1, col_res2 = st.columns([1, 1])
        
        with col_res1:
            st.subheader("Decision Summary")
            if prediction == 1 or prob_fraud >= 0.5:
                st.markdown('<div style="text-align: center; padding: 2rem; border-radius: 0.5rem; background-color: #FEE2E2; border: 1px solid #EF4444;"><span class="badge-fraud" style="font-size: 1.5rem; padding: 0.5rem 1.5rem;">🚨 FRAUDULENT PATTERN DETECTED</span></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="text-align: center; padding: 2rem; border-radius: 0.5rem; background-color: #D1FAE5; border: 1px solid #10B981;"><span class="badge-legit" style="font-size: 1.5rem; padding: 0.5rem 1.5rem;">✅ TRANSACTION LEGITIMATE</span></div>', unsafe_allow_html=True)
                
            st.markdown(f"**Legitimate Probability:** `{prob_legit*100:.2f}%`")
            st.markdown(f"**Fraudulent Probability:** `{prob_fraud*100:.2f}%`")
            st.markdown(f"**Confidence Rating:** `{(max(prob_fraud, prob_legit) * 100):.2f}%`")
            
        with col_res2:
            st.subheader("Feature Attribution (Predictive Explanation)")
            
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
            fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        st.markdown("### Analysis")
        attribution_items = []
        if feat_v14 < -2.0:
            attribution_items.append("⚠️ **V14 Extreme Drop**: V14 is very low, which is associated with fraud.")
        if feat_v17 < -2.0:
            attribution_items.append("⚠️ **V17 Extreme Drop**: V17 is very low, which is associated with fraud.")
        if feat_v4 > 1.5:
            attribution_items.append("⚠️ **V4 Positive Shift**: V4 is elevated, which is associated with fraud.")
        if tx_amount > 1000.0:
            attribution_items.append("⚠️ **Outlier Amount**: The transaction amount is high.")
            
        if not attribution_items:
            st.write("✅ **Attribution Verdict**: No anomalies detected.")
        else:
            for item in attribution_items:
                st.markdown(item)

elif page == "Performance Analytics":
    st.markdown('<h1 class="main-title">Rigorous Performance Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Comprehensive classification statistics, confusion matrices, ROC, and PR curves</p>', unsafe_allow_html=True)
    
    from sklearn.model_selection import train_test_split
    
    df_clean = df.dropna()
    X = df_clean.drop(columns=['Class'])
    y = df_clean['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    X_test_scaled = X_test.copy()
    X_test_scaled[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])
    
    y_pred = model.predict(X_test_scaled)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        dfunc = model.decision_function(X_test_scaled)
        y_prob = 1 / (1 + np.exp(-dfunc))
        
    metrics = evaluate_classifier(y_test, y_pred, y_prob)
    
    col_stat1, col_stat2 = st.columns([1, 1])
    
    with col_stat1:
        st.subheader("1. Active Model Classification Report")
        st.write(f"Loaded Active Model: `{type(model).__name__}`")
        
        # Display classification report cleanly
        rep_df = pd.DataFrame(metrics['class_report']).transpose()
        st.dataframe(rep_df.style.format("{:.4f}"))
        
        st.markdown("""
        ### Understanding Key Metrics:
        - **Precision**: Of all transactions predicted as fraud, what fraction actually was fraud? (Minimizes false alarms and innocent blockings).
        - **Recall (Sensitivity)**: Of all actual fraud, what fraction did the model successfully intercept? (Minimizes financial losses).
        - **F1-Score**: Harmonic mean of Precision and Recall. Highly suited for imbalanced datasets.
        """)
        
    with col_stat2:
        st.subheader("2. Test Confusion Matrix")
        st.plotly_chart(plot_confusion_matrix_plotly(metrics['confusion_matrix']), use_container_width=True)
        
    st.markdown("---")
    st.subheader("3. Model Signal Curves")
    col_curve1, col_curve2 = st.columns(2)
    
    with col_curve1:
        st.plotly_chart(plot_roc_curve_plotly(y_test, y_prob, type(model).__name__), use_container_width=True)
        st.markdown("**ROC-AUC:** Measures the probability that the model ranks a random fraud transaction higher than a random legitimate one. Great general benchmark.")
        
    with col_curve2:
        st.plotly_chart(plot_pr_curve_plotly(y_test, y_prob, type(model).__name__), use_container_width=True)
        st.markdown("**Precision-Recall Curve:** Highly recommended for imbalanced datasets because it focuses entirely on the minority class and doesn't get distorted by large True Negatives.")

elif page == "About & Diagnostics":
    st.markdown('<h1 class="main-title">FraudShield Academic Diagnostics & About</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">System architecture overview, academic overfitting analysis, and research credentials</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("1. Overfitting & Class-Imbalance Analysis")
        st.markdown(get_overfitting_explanation())
        
    with col2:
        st.subheader("2. Student Profile & Credentials")
        st.markdown("""
        <div style="background-color: #F8FAFC; padding: 1.5rem; border-radius: 12px; border: 1px solid #E2E8F0; margin-bottom: 1.5rem;">
            <p style="margin: 0; font-size: 0.85rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">Lead Developer & researcher</p>
            <h3 style="margin: 0.5rem 0 0.25rem 0; font-size: 1.5rem; font-weight: 700; color: #0F172A;">Roshan Perera</h3>
            <p style="margin: 0; font-family: 'JetBrains Mono', monospace; font-size: 0.95rem; font-weight: 600; color: #0EA5E9;">Student ID: S25026203</p>
            <p style="margin: 1rem 0 0 0; font-size: 0.9rem; color: #475569; line-height: 1.5;">Submitted as part of the MSc Advanced Machine Learning & Financial Risk Mitigation Curriculum.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("3. System & Library Diagnostics")
        st.write("- **System Name:** FraudShield")
        st.write("- **Python Version:** 3.10+ (Containerized Environment)")
        st.write("- **Machine Learning framework:** scikit-learn & XGBoost")
        st.write("- **Standard Scaler State:** Fitted, cached in `scaler.pkl`")
        st.write("- **Oversampling Toolkit:** imbalanced-learn / SMOTE ready")
        st.write("- **Web Engine:** Streamlit Core")
        
        st.info("💡 **Academic Note:** When presenting this assignment, emphasize that splitting the dataset must always occur *prior* to oversampling to prevent test set contamination and optimistic bias in performance metrics.")
