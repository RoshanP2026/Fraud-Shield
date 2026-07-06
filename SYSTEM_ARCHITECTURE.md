# FraudShield System Architecture Documentation

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     FraudShield System                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Data Layer │    │  ML Layer    │    │  App Layer   │      │
│  ├──────────────┤    ├──────────────┤    ├──────────────┤      │
│  │ • CSV Files  │    │ • Training   │    │ • Streamlit  │      │
│  │ • Dataset    │    │ • Models     │    │ • Dashboard  │      │
│  │ • Logs       │    │ • Scalers    │    │ • API        │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 2. Data Flow Diagram

```
┌─────────────┐
│ Raw Data    │
│ (CSV)       │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Data        │
│ Cleaning    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Feature     │
│ Engineering │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Scaling     │
│ (Standard)  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Imbalance   │
│ Handling    │
│ (SMOTE)     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Model       │
│ Training    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Evaluation  │
│ & Metrics   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Deployment  │
│ (Save)      │
└─────────────┘
```

## 3. ML Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FraudShield ML Pipeline                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │  Input   │→ │  Clean   │→ │  Encode  │→ │  Scale   │          │
│  │  Data    │  │  Data    │  │  Data    │  │  Data    │          │
│  └──────────┘  └──────────┘  └──────────┘  └────┬─────┘          │
│                                                │                   │
│                                                ▼                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ Feature  │→ │ Imbalance│→ │ Training │→ │ Evaluate │          │
│  │ Engineer │  │ Handling │  │  Models  │  │  Models  │          │
│  └──────────┘  └──────────┘  └────┬─────┘  └────┬─────┘          │
│                                     │              │               │
│                                     └──────┬───────┘               │
│                                            │                       │
│                                            ▼                       │
│                                    ┌──────────┐                   │
│                                    │ Deploy   │                   │
│                                    │ & Save   │                   │
│                                    └──────────┘                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 4. Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Production Deployment                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Client     │    │   Streamlit  │    │   Model      │      │
│  │   Browser    │───→│   Server     │───→│   Server     │      │
│  │              │    │   (app.py)   │    │   (API)      │      │
│  └──────────────┘    └──────┬───────┘    └──────┬───────┘      │
│                              │                   │               │
│                              ▼                   ▼               │
│                     ┌──────────────┐    ┌──────────────┐       │
│                     │   File       │    │   Memory     │       │
│                     │   Storage    │    │   Cache      │       │
│                     │  (Models)    │    │  (Scaler)    │       │
│                     └──────────────┘    └──────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 5. Sequence Diagram - Prediction Flow

```
User          Streamlit      Model        Logger        SHAP
 │                │            │            │            │
 │─Transaction──→│            │            │            │
 │                │            │            │            │
 │                │─Preprocess→│            │            │
 │                │            │            │            │
 │                │─Predict───→│            │            │
 │                │            │            │            │
 │                │←─Result────│            │            │
 │                │            │            │            │
 │                │─Log───────→│            │            │
 │                │            │            │            │
 │                │─Explain──→│            │            │
 │                │            │            │            │
 │                │←─SHAP─────│            │            │
 │                │            │            │            │
 │←─Display───────│            │            │            │
 │                │            │            │            │
```

## 6. Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Component Interactions                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐                                                   │
│  │   app.py    │                                                   │
│  │ (Dashboard) │                                                   │
│  └──────┬──────┘                                                   │
│         │                                                           │
│         ├──→ utils.py (Visualization)                               │
│         │                                                           │
│         ├──→ train.py (Training)                                   │
│         │                                                           │
│         ├──→ explainability.py (SHAP)                              │
│         │                                                           │
│         ├──→ logger.py (Logging)                                    │
│         │                                                           │
│         └──→ pipeline.py (ML Pipeline)                             │
│                  │                                                  │
│                  ├──→ feature_engineering.py                        │
│                  │                                                  │
│                  └──→ saved_models/ (Model Files)                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 7. Module Dependencies

```
FraudShield/
├── Core Modules
│   ├── train.py (depends on: sklearn, xgboost, lightgbm, imbalanced-learn)
│   ├── explainability.py (depends on: shap, sklearn)
│   ├── feature_engineering.py (depends on: pandas, numpy, sklearn)
│   ├── pipeline.py (depends on: sklearn, joblib)
│   └── logger.py (depends on: pandas, csv, json)
│
├── Application
│   ├── app.py (depends on: streamlit, utils)
│   └── app_enhanced.py (depends on: streamlit, all modules)
│
└── Utilities
    └── utils.py (depends on: pandas, plotly, sklearn)
```

## 8. Data Model

```
Transaction Data Structure:
├── Time (float): Seconds from first transaction
├── V1-V28 (float): PCA-transformed features
├── Amount (float): Transaction amount
└── Class (int): 0=Legitimate, 1=Fraudulent

Engineered Features:
├── Transaction_Hour (float): Hour of day
├── Weekend_Transaction (int): Weekend flag
├── Transaction_Velocity (float): Transactions per hour
├── Large_Transaction_Flag (int): High amount flag
├── Amount_to_Average_Ratio (float): Normalized amount
├── Log_Amount (float): Log-transformed amount
├── V14_V17_Interaction (float): PCA interaction
├── V4_V11_Interaction (float): PCA interaction
├── Negative_Feature_Sum (float): Sum of negative features
├── Positive_Feature_Sum (float): Sum of positive features
├── Amount_Risk_Score (float): Amount-based risk
├── PCA_Risk_Score (float): PCA-based risk
└── Distance_From_Home_Simulated (float): Distance metric
```

## 9. API Endpoints (Streamlit)

```
GET /                          → Home Dashboard
GET /predict                   → Prediction Interface
GET /analytics                 → Analytics Dashboard
GET /performance               → Model Performance
GET /error-analysis            → Error Analysis
GET /about                     → About & Diagnostics

POST /predict (internal)       → Make Prediction
POST /train (internal)         → Train Models
GET /logs (internal)           → Get Prediction Logs
```

## 10. Security & Compliance

```
Data Security:
├── Input Validation: All user inputs validated
├── Data Sanitization: Missing values handled
├── Model Security: Models encrypted at rest
└── Audit Trail: All predictions logged

Compliance:
├── GDPR: Data minimization principles
├── Audit Logging: Complete prediction history
├── Model Explainability: SHAP for transparency
└── Error Handling: Graceful degradation
```

## 11. Performance Considerations

```
Optimization Strategies:
├── Caching: Streamlit caching for data and models
├── Lazy Loading: Models loaded on demand
├── Batch Processing: Efficient data handling
├── Parallel Processing: Multi-core training
└── Memory Management: Efficient data structures

Scalability:
├── Horizontal Scaling: Multiple model instances
├── Load Balancing: Request distribution
├── Database Integration: For production deployment
└── API Gateway: For microservices architecture
```

## 12. Monitoring & Observability

```
Metrics Tracked:
├── System Metrics: CPU, Memory, Response Time
├── Business Metrics: Fraud Detection Rate, False Positive Rate
├── Model Metrics: Prediction Confidence, Feature Importance
└── User Metrics: Dashboard Usage, Feature Adoption

Logging:
├── Prediction Logs: Timestamp, Probability, Features
├── Audit Logs: Model loads, predictions, errors
├── System Logs: Application events, errors
└── Performance Logs: Processing time, resource usage
```

## 13. Technology Stack

```
Frontend:
├── Streamlit: Web framework
├── Plotly: Interactive visualizations
└── HTML/CSS: Custom styling

Backend:
├── Python 3.10+: Core language
├── scikit-learn: ML algorithms
├── XGBoost: Gradient boosting
├── LightGBM: Light gradient boosting
├── PyTorch: Neural networks
└── SHAP: Model explainability

Data Processing:
├── pandas: Data manipulation
├── numpy: Numerical computing
└── joblib: Model serialization

Infrastructure:
├── File System: Model storage
├── CSV: Data storage
└── Logging: Audit trail
```

## 14. Development Workflow

```
Development Cycle:
1. Feature Development → Code in respective modules
2. Unit Testing → Test individual components
3. Integration Testing → Test module interactions
4. Dashboard Testing → Test Streamlit interface
5. Performance Testing → Benchmark model performance
6. Documentation → Update README and architecture docs

Version Control:
├── Git: Version control
├── Branching: Feature branches
├── Code Review: Peer review process
└── CI/CD: Automated testing (future)
```

## 15. Future Enhancements

```
Planned Improvements:
├── Real-time API: REST API for production
├── Database Integration: PostgreSQL for data storage
├── Cloud Deployment: AWS/GCP/Azure deployment
├── Advanced Models: Deep learning, ensemble methods
├── Real-time Monitoring: Prometheus/Grafana
├── Automated Retraining: Scheduled model updates
├── A/B Testing: Model comparison framework
└── Mobile App: Native mobile application
```
