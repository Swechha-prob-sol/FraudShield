# FraudShield: End-to-End AI/ML Credit Card Fraud Detection System

FraudShield is an enterprise-grade, real-time credit card fraud detection platform engineered with modern Machine Learning, Deep Learning, FastAPI, Redis, PostgreSQL, React, and MLOps tooling.

---

## 10-Week Development Roadmap

1. **Week 1:** Project Setup, Architecture & Exploratory Data Analysis (EDA)
2. **Week 2:** Baseline Modeling (Logistic Regression & Stratified Evaluation)
3. **Week 3:** Feature Engineering & Temporal Dynamics
4. **Week 4:** Advanced Gradient Boosting (LightGBM) & Extreme Imbalance Handling
5. **Week 5:** Deep Learning Anomaly Detection (Autoencoder), Stacking, Calibration & Explainability (SHAP)
6. **Week 6:** High-Throughput Inference Backend (FastAPI & Pydantic)
7. **Week 7:** In-Memory Streaming & Transaction Replay Simulation (Redis)
8. **Week 8:** Persistence & Analyst Investigation Portal (PostgreSQL & React)
9. **Week 9:** Automated Testing, MLflow Experiment Tracking, Evidently Drift Monitoring
10. **Week 10:** Containerization (Docker), Production Deployment & Live Demo

---

## Repository Structure

```text
FraudShield/
├── .gitignore               # Strict exclusion rules (data, credentials, artifacts)
├── README.md                # Project documentation and roadmap
├── requirements.txt         # Pinned production and research dependencies
├── config/                  # Configuration files (YAML, settings)
├── data/                    # Dataset directory (Ignored by Git)
│   ├── raw/                 # Original, immutable transaction records
│   └── processed/           # Transformed, scaled, and split arrays/dataframes
├── models/                  # Serialized model weights and scalers (Ignored by Git)
├── notebooks/               # Jupyter exploration and experimental notebooks
├── src/                     # Core production source code
│   ├── data/                # Ingestion, validation, and loaders
│   ├── features/            # Preprocessors, transformations, scaling
│   ├── models/              # Estimator architectures, training, evaluation
│   └── utils/               # Logging, metrics, config loaders
└── tests/                   # Automated unit and integration test suite
```

---

## Getting Started

### 1. Create and Activate Virtual Environment
```bash
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
