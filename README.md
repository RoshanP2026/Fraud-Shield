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

```
FraudShield/
│
├── CreditCardFraudDetection/
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
pip install -r CreditCardFraudDetection/requirements.txt
```

---

# Running the Application

Launch the Streamlit application:

```bash
streamlit run CreditCardFraudDetection/app.py
```

The application will be available at:

```
http://localhost:8501
```

---

# Troubleshooting

### Missing Model Files

If the trained model files are unavailable, regenerate them using:

```bash
python CreditCardFraudDetection/train_model.py
```

---

### Port Already in Use

Run the application on another port:

```bash
streamlit run CreditCardFraudDetection/app.py --server.port 8502
```

---

### Dependency Issues

Upgrade pip and reinstall the required packages:

```bash
pip install --upgrade pip
pip install -r CreditCardFraudDetection/requirements.txt --force-reinstall
```

---

# Future Enhancements

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
