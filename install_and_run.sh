#!/bin/bash
# install_and_run.sh
# Automated dependency installer and runtime manager for Credit Card Fraud Detection

# Ensure pip3 is available in the container
if ! command -v pip3 &> /dev/null
then
    echo "[SYSTEM] pip3 is not available. Triggering system-level package installation..."
    apt-get update && apt-get install -y python3-pip python3-venv
fi

# Install Python ML requirements
echo "[SYSTEM] Installing academic and production ML packages from requirements.txt..."
pip3 install --no-cache-dir --break-system-packages -r CreditCardFraudDetection/requirements.txt

# Execute training script to generate initial pkl artifacts if not present
if [ ! -f CreditCardFraudDetection/model.pkl ] || [ ! -f CreditCardFraudDetection/scaler.pkl ]; then
    echo "[SYSTEM] Pre-trained models not found. Triggering ML training pipeline..."
    python3 CreditCardFraudDetection/train_model.py
fi

# Launch Streamlit app on the designated port (3000)
echo "[SYSTEM] Launching Streamlit web application on port 3000..."
streamlit run CreditCardFraudDetection/app.py --server.port 3000 --server.address 0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false
