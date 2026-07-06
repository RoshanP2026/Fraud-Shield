<<<<<<< HEAD
<<<<<<< HEAD
# FraudShield - Advanced Credit Card Fraud Detection System

<div align="center">
  <h3>MSc Advanced Machine Learning Project</h3>
  <p>Real-time fraud detection using Machine Learning with SMOTE and ensemble methods</p>
</div>

## Overview

FraudShield is a comprehensive credit card fraud detection system that addresses the challenge of severe class imbalance in financial transaction data. The system uses advanced ML techniques including SMOTE (Synthetic Minority Over-sampling Technique), multiple classifier models, and rigorous evaluation metrics optimized for imbalanced datasets.

### Key Features

- **Interactive Web Interface**: Built with Streamlit for real-time fraud prediction and analysis
- **Multiple ML Models**: Logistic Regression, Random Forest, and XGBoost/Gradient Boosting
- **SMOTE Integration**: Handles class imbalance through synthetic oversampling
- **Comprehensive Analytics**: ROC curves, Precision-Recall curves, confusion matrices, and feature analysis
- **Real-time Prediction**: Interactive transaction risk evaluation with scenario templates
- **Academic Diagnostics**: Overfitting analysis and model performance metrics

## Project Structure

```
FraudShield/
├── CreditCardFraudDetection/
│   ├── app.py              # Main Streamlit application
│   ├── utils.py            # Utility functions for data processing and visualization
│   ├── train_model.py      # ML training pipeline script
│   ├── requirements.txt    # Python dependencies
│   ├── notebook.ipynb      # Jupyter notebook for analysis
│   ├── logo.jpg            # Application logo
│   ├── banner.jpg          # Application banner
│   └── favicon.png         # Favicon
├── README.md               # This file
├── .gitignore              # Git ignore rules
└── install_and_run.sh      # Installation and run script
```

## Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Git (for cloning the repository)

## Installation

### Option 1: Using the installation script (Linux/Mac/WSL)

```bash
chmod +x install_and_run.sh
./install_and_run.sh
```

### Option 2: Manual installation

1. **Clone the repository**
```bash
git clone <your-repository-url>
cd FraudShield
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r CreditCardFraudDetection/requirements.txt
```

4. **Train the initial model** (optional - will be done automatically on first run)
```bash
python CreditCardFraudDetection/train_model.py
```

## Running the Application

### Using the installation script
```bash
./install_and_run.sh
```

### Manual execution
```bash
streamlit run CreditCardFraudDetection/app.py
```

The application will start and be available at `http://localhost:8501`

## Application Pages

1. **Home**: System overview and dashboard with key statistics
2. **Dataset Explorer**: Interactive data inspection and analysis
3. **Exploratory Data Analysis**: Visualizations for amount, time, and feature distributions
4. **Model Training**: Train and optimize ML models with custom parameters
5. **Prediction System**: Real-time transaction fraud risk evaluation
6. **Performance Analytics**: Comprehensive model performance metrics and visualizations
7. **About & Diagnostics**: System information and academic analysis

## Dataset

The system uses a synthetic credit card fraud dataset that mimics the Kaggle Credit Card Fraud dataset structure:
- **Time**: Seconds elapsed since the first transaction
- **V1-V28**: PCA-transformed features (principal components)
- **Amount**: Transaction amount
- **Class**: Target variable (0 = Legitimate, 1 = Fraudulent)

The dataset is automatically generated on first run if not present.

## Model Performance

The system evaluates models using metrics optimized for imbalanced datasets:
- **Precision**: Minimizes false alarms
- **Recall**: Maximizes fraud detection
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Overall classification performance
- **Precision-Recall AUC**: Performance on minority class

## Academic Context

This project was developed as part of an MSc Advanced Machine Learning curriculum focusing on:
- Class imbalance handling techniques
- Ensemble learning methods
- Model evaluation in imbalanced scenarios
- Overfitting detection and prevention

**Important Note**: SMOTE is applied ONLY to the training set to prevent test set contamination and ensure valid performance metrics.

## Developer

- **Name**: Roshan Perera
- **Student ID**: S25026203
- **Program**: MSc Advanced Machine Learning & Financial Risk Mitigation

## Troubleshooting

### Model files not found
If you see errors about missing model.pkl or scaler.pkl files:
```bash
python CreditCardFraudDetection/train_model.py
```

### Port already in use
If port 8501 is in use, specify a different port:
```bash
streamlit run CreditCardFraudDetection/app.py --server.port 8502
```

### Dependency issues
Ensure you have the latest pip and try:
```bash
pip install --upgrade pip
pip install -r CreditCardFraudDetection/requirements.txt --force-reinstall
```

## License

This project is submitted as part of academic coursework. Please contact the author for usage permissions.

## Acknowledgments

- Kaggle Credit Card Fraud Dataset (original dataset structure reference)
- scikit-learn, imbalanced-learn, and Streamlit communities
=======
# CreditCardFraudDetector_test
Check121tst
>>>>>>> 3b0dca6e0592a9191d4db0e165d202656840b37d
=======
### Fraud-Shield — Credit Card Fraud Detection System

Fraud-Shield is a machine learning project that detects fraudulent credit card transactions using classification algorithms. The goal is to improve transaction security by identifying suspicious activities with high accuracy and recall.

### Features

* Fraud detection using machine learning algorithms
* Data preprocessing and feature scaling
* Handling class imbalance in transaction data
* Model evaluation using Confusion Matrix
* Performance analysis using Classification Report
* Visualization of fraud detection results

### Project Structure



### Dataset

The dataset contains historical credit card transactions with features representing transaction characteristics and a target variable indicating whether the transaction is fraudulent (1) or legitimate (0).

### Technologies Used

Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Jupyter Notebook / Google Colab

### Evaluation Metrics

The performance of the fraud detection model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Classification Report

### How to Run

1
Install dependencies
Copy

2
Run the project
Copy

3
Open the notebook
Copy

### Example Output

Metric

Value

Accuracy
99.8%

Precision
97.5%

Recall
94.2%

F1-Score
95.8%

### Future Improvements

* Deploy the model as a web API
* Add real-time transaction monitoring
* Integrate deep learning techniques
* Implement anomaly detection methods
* Build an interactive dashboard for fraud analysis

### Author

Roshan Perera
ID : S25026203

🛡️ “Fraud-Shield — Protecting Every Transaction with Intelligent Detection.”
>>>>>>> 978739c2a8618396bd3c0ed773332c335140f94d
