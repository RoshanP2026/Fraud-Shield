<<<<<<< Updated upstream
# FraudShield – Credit Card Fraud Detection System

<div align="center">

## MSc Advanced Machine Learning Project

**An Intelligent Credit Card Fraud Detection System Using Machine Learning and SMOTE**

<img src="favicon.png" alt="FraudShield Logo" width="220">

</div>

---

# Overview

FraudShield is a machine learning-based application developed to identify fraudulent credit card transactions. The project addresses one of the major challenges in fraud detection—**highly imbalanced datasets**—by applying the Synthetic Minority Over-sampling Technique (SMOTE) together with supervised machine learning algorithms.

The system provides an interactive web application developed using **Streamlit**, enabling users to explore the dataset, train machine learning models, evaluate model performance, and predict the likelihood of fraudulent transactions in real time.

This project was completed as part of the **MSc Advanced Machine Learning** programme and demonstrates the practical application of machine learning techniques in financial risk management.

---

# Project Objectives

The primary objectives of this project are to:

- Develop a machine learning solution for detecting fraudulent credit card transactions.
- Address class imbalance using SMOTE.
- Compare the performance of multiple classification algorithms.
- Evaluate models using metrics suitable for imbalanced classification problems.
- Provide an interactive interface for model training and fraud prediction.
- Demonstrate best practices in machine learning model development and evaluation.

---

# Key Features

- Interactive web application built with Streamlit
- Automated data preprocessing and feature scaling
- Class imbalance handling using SMOTE
- Multiple machine learning classification models
- Real-time fraud prediction
- Model comparison dashboard
- Exploratory Data Analysis (EDA)
- Performance evaluation using appropriate classification metrics
- Confusion Matrix and ROC Curve visualisation
- Overfitting diagnostics and model comparison

---

# Machine Learning Workflow

The project follows a complete machine learning pipeline consisting of:

1. Data Loading
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Scaling
5. Train-Test Split
6. SMOTE Oversampling (Training Set Only)
7. Model Training
8. Model Evaluation
9. Fraud Prediction
10. Performance Analysis

---

# Project Structure
=======
# FraudShield - Advanced Credit Card Fraud Detection System

<div align="center">
  <h3>MSc Advanced Machine Learning Project</h3>
  <p>Enterprise-grade real-time fraud detection with Explainable AI and Advanced ML Techniques</p>
</div>

## 🎯 Overview

FraudShield is a comprehensive, enterprise-grade credit card fraud detection system that addresses the challenge of severe class imbalance in financial transaction data. The system implements state-of-the-art machine learning techniques including multiple algorithms, hyperparameter tuning, explainable AI (SHAP), and comprehensive evaluation metrics optimized for imbalanced datasets.

## ✨ Key Features

### Machine Learning Capabilities
- **Multiple ML Models**: Logistic Regression, Decision Tree, Random Forest, Balanced Random Forest, XGBoost, LightGBM, Gradient Boosting, Neural Network (MLP)
- **Hyperparameter Tuning**: GridSearchCV and RandomizedSearchCV with before/after comparison
- **Class Imbalance Handling**: SMOTE, Random Under Sampling, Balanced Random Forest, Class Weights
- **Cross-Validation**: 5-fold and 10-fold CV with mean accuracy and standard deviation
- **Feature Engineering**: Transaction velocity, risk scores, interaction features, time-based features

### Explainable AI
- **SHAP Integration**: Feature attribution and model interpretability
- **Individual Prediction Explanations**: Understand why specific transactions are flagged
- **Summary Plots**: Global feature importance visualization
- **Dependence Plots**: Feature interaction analysis

### System Features
- **ML Pipeline**: End-to-end pipeline (Cleaning → Encoding → Scaling → Feature Engineering → Training → Evaluation → Deployment)
- **Real-time Prediction**: Interactive transaction risk evaluation with scenario templates
- **Comprehensive Logging**: Prediction logs with timestamps, probabilities, and audit trail
- **Model Persistence**: joblib/pickle for production-ready model saving/loading

### Visualization & Analytics
- **Professional Dashboard**: Streamlit-based with multiple tabs (Home, Predict, Analytics, Model Performance, Error Analysis, About)
- **Advanced Visualizations**: ROC curves, Precision-Recall curves, confusion matrices, correlation heatmaps
- **Error Analysis**: False positive/negative analysis with explanations
- **Probability Gauges**: Visual confidence intervals and risk indicators

## 📁 Project Structure
>>>>>>> Stashed changes

```
FraudShield/
│
├── CreditCardFraudDetection/
<<<<<<< Updated upstream
│   ├── app.py
│   ├── train_model.py
│   ├── utils.py
│   ├── notebook.ipynb
│   ├── requirements.txt
│   ├── logo.jpg
│   ├── banner.jpg
│   └── favicon.png
│
├── README.md
├── .gitignore
├── install_and_run.sh
└── favicon.png
```

---

# Technologies Used

## Programming Language

- Python 3.10+

## Machine Learning Libraries

- Scikit-learn
- Imbalanced-learn (SMOTE)
- XGBoost
- NumPy
- Pandas

## Data Visualisation

- Matplotlib
- Seaborn
- Plotly

## Web Application

- Streamlit

---

# Dataset

The project uses a credit card transaction dataset containing both legitimate and fraudulent transactions.

The dataset consists of:

- **Time** – Time elapsed since the first transaction
- **Amount** – Transaction value
- **V1–V28** – PCA-transformed features
- **Class**
  - 0 = Legitimate Transaction
  - 1 = Fraudulent Transaction

Because fraudulent transactions represent only a very small proportion of the data, the dataset is highly imbalanced, making fraud detection a challenging classification problem.

---

# Data Pre-processing

The following preprocessing steps are performed before model training:

- Missing value validation
- Feature scaling
- Train-test split
- Class imbalance analysis
- SMOTE oversampling (training data only)

Applying SMOTE exclusively to the training dataset prevents information leakage and ensures that the evaluation metrics remain reliable.

---

# Machine Learning Models

The following supervised learning algorithms are implemented:

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting
- XGBoost (optional)

Each model is trained using identical training data and evaluated using the same testing dataset to allow a fair performance comparison.

---

# Model Evaluation

Since fraud detection is an imbalanced classification problem, model evaluation extends beyond overall accuracy.

The following performance metrics are used:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Precision-Recall Curve
- Confusion Matrix

Special attention is given to **Recall**, as correctly identifying fraudulent transactions is generally more important than maximising overall accuracy.

---

# Application Modules

The Streamlit application includes the following modules:

### Home

Provides an overview of the project and key statistics.

### Dataset Explorer

Allows users to inspect the dataset interactively.

### Exploratory Data Analysis

Visualises transaction distributions, feature relationships, and fraud patterns.

### Model Training

Enables users to train different machine learning models and compare their performance.

### Prediction System

Allows users to enter transaction information and receive an instant fraud prediction.

### Performance Analytics

Displays evaluation metrics, ROC curves, confusion matrices, and model comparisons.

### About & Diagnostics

Provides project information together with overfitting analysis and diagnostic reports.

---

# Installation

Clone the repository:

=======
│   ├── app.py                  # Main Streamlit application (original)
│   ├── app_enhanced.py         # Enhanced Streamlit application with all features
│   ├── utils.py                # Utility functions for data processing and visualization
│   ├── train_model.py          # Original ML training pipeline
│   ├── requirements.txt        # Python dependencies
│   ├── notebook.ipynb          # Jupyter notebook for analysis
│   ├── logo.jpg                # Application logo
│   ├── banner.jpg              # Application banner
│   ├── favicon.png             # Favicon
│   ├── model.pkl               # Trained model
│   └── scaler.pkl              # Fitted scaler
├── dataset/                    # Dataset directory
├── models/                     # Saved models directory
├── images/                     # Images directory
├── notebooks/                  # Jupyter notebooks directory
├── saved_models/               # Production-ready saved models
├── logs/                       # Prediction and audit logs
├── train.py                    # Advanced training module with multiple models
├── explainability.py           # SHAP-based explainable AI module
├── feature_engineering.py      # Advanced feature engineering module
├── pipeline.py                 # End-to-end ML pipeline module
├── logger.py                   # Logging and audit trail module
├── requirements.txt            # Complete dependencies
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
└── install_and_run.sh          # Installation and run script
```

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)
- Git (for cloning the repository)

### Option 1: Using the installation script (Linux/Mac/WSL)
>>>>>>> Stashed changes
```bash
git clone https://github.com/RoshanP2026/Fraud-Shield.git
```

Navigate into the project directory:

```bash
cd Fraud-Shield
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment.

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

<<<<<<< Updated upstream
---

# Running the Application

Launch the Streamlit application:

=======
4. **Train models** (optional - will use fallback if not trained)
```bash
python train.py
```

## 🎮 Running the Application

### Enhanced Dashboard (Recommended)
```bash
streamlit run CreditCardFraudDetection/app_enhanced.py
```

### Original Dashboard
>>>>>>> Stashed changes
```bash
streamlit run CreditCardFraudDetection/app.py
```

The application will be available at:

<<<<<<< Updated upstream
```
http://localhost:8501
```

---

# Troubleshooting
=======
## 📊 Application Pages

### Enhanced Dashboard Pages
1. **🏠 Home**: System overview, dashboard statistics, and recent activity
2. **🔮 Predict**: Real-time transaction prediction with SHAP explanations and probability gauges
3. **📊 Analytics**: Comprehensive EDA with distribution analysis, correlation, and feature separation
4. **🎯 Model Performance**: ROC curves, PR curves, confusion matrices, and classification reports
5. **🔬 Error Analysis**: False positive/negative analysis with detailed explanations
6. **ℹ️ About & Diagnostics**: System architecture, developer profile, and technical specifications

### Training Module Features
Run `python train.py` to access:
- Multiple model training with comparison tables
- Hyperparameter tuning with before/after metrics
- Class imbalance technique comparison
- Cross-validation analysis
- Automatic model saving

## 📈 Dataset
>>>>>>> Stashed changes

### Missing Model Files

If the trained model files are unavailable, regenerate them using:

<<<<<<< Updated upstream
=======
## 🎯 Model Performance Metrics

The system evaluates models using metrics optimized for imbalanced datasets:
- **Accuracy**: Overall classification accuracy
- **Precision**: Minimizes false alarms
- **Recall**: Maximizes fraud detection
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Overall classification performance
- **Precision-Recall AUC**: Performance on minority class

## 🔬 Academic Context

This project was developed as part of an MSc Advanced Machine Learning curriculum focusing on:
- Class imbalance handling techniques (SMOTE, undersampling, class weights)
- Ensemble learning methods (Random Forest, XGBoost, LightGBM)
- Hyperparameter optimization (GridSearchCV, RandomizedSearchCV)
- Model explainability (SHAP)
- Model evaluation in imbalanced scenarios
- Overfitting detection and prevention
- Feature engineering for fraud detection
- End-to-end ML pipeline development

**Important Note**: SMOTE is applied ONLY to the training set to prevent test set contamination and ensure valid performance metrics.

## 👨‍💻 Developer

- **Name**: Roshan Perera
- **Student ID**: S25026203
- **Program**: MSc Advanced Machine Learning & Financial Risk Mitigation

## 🛠️ Troubleshooting

### Model files not found
If you see errors about missing model files:
>>>>>>> Stashed changes
```bash
python train.py
```

---

### Port Already in Use

Run the application on another port:

```bash
streamlit run CreditCardFraudDetection/app_enhanced.py --server.port 8502
```

---

### Dependency Issues

Upgrade pip and reinstall the required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

<<<<<<< Updated upstream
---
=======
## 📝 License
>>>>>>> Stashed changes

# Future Enhancements

<<<<<<< Updated upstream
Possible future improvements include:

- Deep learning models for fraud detection
- Real-time transaction streaming
- REST API deployment
- Docker containerisation
- Cloud deployment using AWS or Azure
- Automated hyperparameter optimisation
- Explainable AI (SHAP/LIME)
- Continuous model retraining

---

# Author

**Roshan Perera**

**Student ID:** S25026203

---

# Acknowledgements

This project makes use of the following open-source libraries:

- Scikit-learn
- Imbalanced-learn
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Plotly
- XGBoost

The project is inspired by the structure of the publicly available Credit Card Fraud Detection dataset originally published on Kaggle.

---

# License

This repository has been developed solely for academic purposes as part of an MSc programme.

It may be used for learning and research with appropriate acknowledgement to the author.

---

<div align="center">

**FraudShield**

*Machine Learning for Intelligent Credit Card Fraud Detection*

</div>
=======
## 🙏 Acknowledgments

- Kaggle Credit Card Fraud Dataset (original dataset structure reference)
- scikit-learn, imbalanced-learn, XGBoost, LightGBM, SHAP, and Streamlit communities
- Open-source ML ecosystem contributors

## 📚 Key Technologies

- **ML Frameworks**: scikit-learn, XGBoost, LightGBM, PyTorch
- **Data Processing**: pandas, numpy
- **Explainability**: SHAP
- **Visualization**: plotly, matplotlib, seaborn
- **Web Framework**: Streamlit
- **Imbalance Handling**: imbalanced-learn (SMOTE)
- **Optimization**: Optuna, scikit-optimize
>>>>>>> Stashed changes
