# FraudShield - Credit Card Fraud Detection System

A machine learning system for detecting fraudulent credit card transactions in real-time.

## What This App Does

FraudShield analyzes credit card transaction data to identify potentially fraudulent activities. It uses machine learning models trained on historical transaction data to assess the risk level of new transactions and flag suspicious ones for review.

The main challenge in fraud detection is that fraudulent transactions are very rare compared to legitimate ones (typically less than 1% of all transactions). This app addresses that imbalance using specialized techniques.

## How to Use

### Installation

1. Make sure you have Python 3.10 or higher installed
2. Install the required packages:
```bash
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## App Sections

### Home
Shows overall statistics about the dataset and recent prediction activity. You can see total transactions, fraud cases, and the proportion of fraudulent transactions.

### Predict
This is the main feature. Enter transaction details (amount, time, and key PCA features) to get a fraud risk assessment. The app will:
- Predict whether the transaction is legitimate or fraudulent
- Show the confidence level as a percentage
- Display a risk gauge visualization
- Explain which features contributed most to the decision

There are also preset scenarios you can use to see how the system handles different types of transactions.

### Analytics
Explore the data through visualizations:
- See how transaction amounts and times are distributed
- View correlations between features
- Analyze which features best separate fraudulent from legitimate transactions

### Model Performance
View how well the trained model performs:
- Accuracy, precision, recall, and F1-score
- Confusion matrix showing correct vs incorrect predictions
- ROC curve and Precision-Recall curve

### Error Analysis
Look at cases where the model made mistakes:
- False positives: legitimate transactions flagged as fraud
- False negatives: fraudulent transactions that were missed
- Common characteristics of these errors

### About
Developer information and credentials.

## The Data

The system uses credit card transaction data with:
- **Time**: Seconds elapsed since the first transaction
- **V1-V28**: PCA-transformed features (these are anonymized principal components from the original data)
- **Amount**: Transaction amount in dollars
- **Class**: 0 for legitimate, 1 for fraudulent

The PCA features (V1-V28) are the result of dimensionality reduction applied to the original transaction data for privacy protection. Some of these features (like V14, V17, V4, V11) are particularly important for distinguishing fraud from legitimate transactions.

## Technical Details

### Models Used
- Random Forest (primary model)
- Logistic Regression
- XGBoost
- Other ensemble methods

### Handling Imbalanced Data
Since fraud cases are rare, the system uses techniques like:
- SMOTE (Synthetic Minority Over-sampling Technique) to create synthetic fraud examples
- Class weights to give more importance to fraud cases during training
- Specialized evaluation metrics (precision, recall, F1) instead of just accuracy

### Evaluation Metrics
- **Precision**: Of all transactions flagged as fraud, how many were actually fraud?
- **Recall**: Of all actual fraud cases, how many did we catch?
- **F1-Score**: Balance between precision and recall
- **ROC-AUC**: Overall ability to distinguish between classes

## Author

**Roshan Perera**  
Student ID: S25026203  
MSc Advanced Machine Learning & Financial Risk Mitigation
