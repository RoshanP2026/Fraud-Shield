# 🛡️ FraudShield - Advanced Credit Card Fraud Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

### MSc Advanced Machine Learning Project

**An Explainable AI-Powered Credit Card Fraud Detection System Using Machine Learning and SMOTE**

<img src="favicon.png" width="180">

</div>

---

# 📖 Overview

FraudShield is an enterprise-grade machine learning application designed to detect fraudulent credit card transactions with high accuracy.

The project addresses one of the biggest challenges in fraud detection—**extreme class imbalance**—by applying the **Synthetic Minority Over-sampling Technique (SMOTE)** alongside advanced supervised learning algorithms.

The application enables users to:

- Explore the dataset
- Train multiple ML models
- Compare model performance
- Perform real-time fraud prediction
- Visualize evaluation metrics
- Interpret predictions using Explainable AI

The system is deployed as an interactive **Streamlit web application**.

---

# 🌐 Live Demo

**Application**

https://fraud-shield-s25026203.streamlit.app

**GitHub Repository**

https://github.com/RoshanP2026/Fraud-Shield

---

# ✨ Key Features

- Interactive Streamlit Dashboard
- Credit Card Fraud Detection
- Multiple Machine Learning Models
- SMOTE Class Balancing
- Hyperparameter Tuning
- Feature Scaling
- Model Comparison Dashboard
- Exploratory Data Analysis
- Real-Time Prediction
- Explainable AI (SHAP Ready)
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Overfitting Detection
- Cross Validation
- Model Persistence

---

# 🧠 Machine Learning Pipeline

```
Dataset
    │
    ▼
Data Cleaning
    │
    ▼
Exploratory Data Analysis
    │
    ▼
Feature Engineering
    │
    ▼
Feature Scaling
    │
    ▼
Train/Test Split
    │
    ▼
SMOTE Oversampling
    │
    ▼
Model Training
    │
    ▼
Model Evaluation
    │
    ▼
Fraud Prediction
    │
    ▼
Streamlit Dashboard
```

---

# 📂 Project Structure

```
FraudShield/

│
├── CreditCardFraudDetection/
│   ├── app.py
│   ├── train_model.py
│   ├── utils.py
│   ├── notebook.ipynb
│   ├── requirements.txt
│   ├── banner.jpg
│   ├── logo.jpg
│   └── favicon.png
│
├── dataset/
├── models/
├── screenshots/
├── README.md
├── .gitignore
└── install_and_run.sh
```

---

# ⚙️ Technologies Used

## Programming Language

- Python 3.10+

## Machine Learning

- Scikit-learn
- Imbalanced-learn (SMOTE)
- XGBoost
- NumPy
- Pandas

## Data Visualization

- Matplotlib
- Plotly
- Seaborn

## Web Framework

- Streamlit

---

# 📊 Dataset

The project uses a real-world credit card transaction dataset consisting of:

- Time
- Amount
- PCA Features (V1 – V28)
- Class

Where

- **0 → Legitimate Transaction**
- **1 → Fraudulent Transaction**

Only around **0.17%** of the transactions are fraudulent, making this a highly imbalanced classification problem.

---

# ⚖️ Why SMOTE?

Fraud datasets are naturally imbalanced.

Without balancing:

- The model becomes biased toward legitimate transactions.
- Fraudulent transactions are often ignored.

FraudShield applies **SMOTE** only to the **training dataset** after the train-test split to prevent data leakage while improving minority-class detection.

---

# 🤖 Machine Learning Models

The project compares several supervised learning algorithms.

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost

Each model is trained using identical preprocessing steps to ensure a fair comparison.

---

# 📈 Model Evaluation Metrics

Fraud detection requires more than simple accuracy.

The following metrics are used:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Precision-Recall AUC
- Confusion Matrix

Special emphasis is placed on **Recall**, since missing fraudulent transactions can lead to significant financial losses.

---

# 📋 Example Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| Logistic Regression | XX% | XX% | XX% | XX% | XX% |
| Random Forest | XX% | XX% | XX% | XX% | XX% |
| Gradient Boosting | XX% | XX% | XX% | XX% | XX% |
| XGBoost | XX% | XX% | XX% | XX% | XX% |

*(Replace with your actual results.)*

---

# 💻 Application Modules

## 🏠 Home

Project overview and dashboard.

## 📊 Dataset Explorer

Interactive dataset inspection.

## 📈 Exploratory Data Analysis

Visualizations and fraud statistics.

## 🤖 Model Training

Train and compare machine learning models.

## 🔍 Prediction

Predict whether a transaction is fraudulent.

## 📉 Performance Dashboard

Confusion Matrix

ROC Curve

Precision-Recall Curve

Classification Report

## ℹ️ About

Project information and technical details.

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/RoshanP2026/Fraud-Shield.git
```

Navigate to the project

```bash
cd Fraud-Shield
```

Create a virtual environment

```bash
python -m venv venv
```

Activate it

Windows

```bash
venv\Scripts\activate
```

Linux/macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

Launch Streamlit

```bash
streamlit run CreditCardFraudDetection/app.py
```

The application will be available at

```
http://localhost:8501
```

---

# 🛠 Troubleshooting

## Missing Model Files

Retrain the model

```bash
python train_model.py
```

## Port Already in Use

```bash
streamlit run CreditCardFraudDetection/app.py --server.port 8502
```

## Dependency Issues

```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

---

# 🔮 Future Improvements

- Deep Learning Models
- LSTM Fraud Detection
- Explainable AI Dashboard
- REST API
- Docker Deployment
- Kubernetes Support
- AWS Cloud Deployment
- Real-Time Transaction Streaming
- Continuous Model Retraining
- AutoML Integration

---

# 👨‍💻 Author

**Roshan Perera**

**Student ID:** S25026203

---

# 🙏 Acknowledgements

This project was developed using the following open-source technologies:

- Scikit-learn
- Streamlit
- Imbalanced-learn
- XGBoost
- Pandas
- NumPy
- Matplotlib
- Plotly
- Seaborn

Special thanks to the open-source community for providing the tools and resources that made this project possible.

---

# 📄 License

This project has been developed solely for academic purposes as part of an MSc programme.

It may be used for learning and research with appropriate acknowledgment to the author.

---

<div align="center">

**FraudShield**

*Machine Learning for Intelligent Credit Card Fraud Detection*

</div>
