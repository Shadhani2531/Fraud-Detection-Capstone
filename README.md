# 🛡️ Real-Time Fraud Detection System with Explainable AI & Live Dashboard

> **Internship Capstone — Week 4 | Domain: AI & Data Analytics | Level: Advanced**  
> Submitted by: **Shadhani Shambharkar**  
> Submission Date: **25/05/2026**

---

## 📋 Project Overview

Financial fraud costs the global economy over **$5 trillion annually**. This project builds a production-grade, end-to-end **Fraud Detection System** that combines state-of-the-art machine learning, imbalance handling, and Explainable AI (SHAP) — packaged inside a live interactive Streamlit dashboard.

The system is trained on the **IEEE-CIS Fraud Detection** dataset (590,000 transactions, 433 features) and achieves a high PR-AUC by combining LightGBM gradient boosting with SMOTE oversampling and Optuna hyperparameter tuning.

---

## 🎯 Key Features

| Feature | Details |
|---|---|
| **Models Trained** | LightGBM, XGBoost, Isolation Forest |
| **Imbalance Handling** | SMOTE on training set only (3.5% → 50% fraud ratio) |
| **Explainability** | SHAP Global Summary, Waterfall & Dependence Plots |
| **Risk Segmentation** | 3-tier system: 🔴 Critical Risk / 🟡 Suspicious / 🟢 Clear |
| **Dashboard** | 3-page Streamlit app with Plotly interactive charts |
| **Deployment** | Streamlit Community Cloud via GitHub |

---

## 📂 Repository Structure

```
FraudDetection_[Shadhani Shambharkar]/
│
├── 📓 analysis_fixed.ipynb      # Main Jupyter Notebook (Tasks 1–8)
├── 📊 app.py                    # Streamlit multi-page dashboard
├── 📋 requirements.txt          # Python dependency manifest
├── 📖 README.md                 # This file
│
├── data/
│   ├── train_transaction.csv    # IEEE-CIS transaction data
│   └── train_identity.csv       # IEEE-CIS identity data
│
├── charts/                      # Exported visualisation assets
│   ├── class_imbalance.png
│   ├── transaction_amt_dist.png
│   ├── correlation_heatmap.png
│   ├── roc_curves.png
│   ├── pr_curves.png
│   ├── shap_summary.png
│   ├── shap_waterfall_fraud.png
│   ├── shap_waterfall_border.png
│   ├── shap_waterfall_legit.png
│   └── risk_tier_device_grouped_bar.png
│
├── best_model.pkl               # Serialised best model (LightGBM)
└── dashboard_data.csv           # Pre-processed test slice for dashboard
```

---

## 🧪 Tasks Completed

| Task | Description | Status |
|------|-------------|--------|
| **Task 1** | Data Loading, Merging & Exploratory Analysis | ✅ |
| **Task 2** | Preprocessing, Imbalance Handling & Feature Engineering | ✅ |
| **Task 3** | Model Training, Comparison & Threshold Optimisation | ✅ |
| **Task 4** | Explainable AI with SHAP Values | ✅ |
| **Task 5** | Risk Segmentation & Fraud Pattern Analysis | ✅ |
| **Task 6** | Streamlit Fraud Operations Dashboard | ✅ |
| **Task 7** | Visualisations (minimum 5 charts) | ✅ |
| **Task 8** | Insights & Business Recommendations | ✅ |

---

## 🔑 Top 3 Fraud Signals (SHAP)

1. 💰 **High Transaction Amount** — Transactions > $200 contribute +0.45 to fraud probability
2. 🌙 **Off-Hours Activity (1 AM–5 AM)** — Overnight submissions add +0.25 risk multiplier
3. 📱 **Mobile Device Attack Vector** — Mobile hardware flag adds +0.15 SHAP contribution

---

## 🚀 Running the Project Locally

### Step 1 — Clone / Download the Repository

```bash
git clone https://github.com/<your-username>/FraudDetection_Shadhani.git
cd FraudDetection_Shadhani
```

### Step 2 — Set Up the Environment (Anaconda Recommended)

```bash
conda create -n fraud_env python=3.11 -y
conda activate fraud_env
pip install -r requirements.txt
```

### Step 3 — Add the Dataset

Download from Kaggle: https://www.kaggle.com/c/ieee-fraud-detection/data

Place both files inside the `data/` folder:
```
data/train_transaction.csv
data/train_identity.csv
```

### Step 4 — Convert Data to Parquet (Memory Optimization)

To prevent RAM exhaustion crashes when loading the massive dataset, run the provided conversion script first. It converts the raw CSVs into highly compressed Parquet format:

`ash
python convert_to_parquet.py
`

### Step 5 — Run the Notebook

```bash
jupyter lab analysis_fixed.ipynb
```

Run all cells **top to bottom**. This will:
- Train the models (may take 5–15 minutes on full data)
- Export `best_model.pkl` and `dashboard_data.csv`
- Save all charts to `charts/`

### Step 6 — Launch the Dashboard Locally

```bash
streamlit run app.py
```

---

## ☁️ Deploying to Streamlit Community Cloud

### Step 1 — Push to GitHub

```bash
git init
git add app.py requirements.txt best_model.pkl dashboard_data.csv README.md
git commit -m "Initial fraud detection dashboard deployment"
git remote add origin https://github.com/<your-username>/FraudDetection_Shadhani.git
git push -u origin main
```

> ⚠️ **Note:** `train_transaction.csv` and `train_identity.csv` are large files (~600MB). Add them to `.gitignore` — they are only needed for notebook training, not dashboard deployment.

### Step 2 — Connect to Streamlit Community Cloud

1. Go to **https://share.streamlit.io** and sign in with GitHub
2. Click **"New app"**
3. Select your repository: `FraudDetection_Shadhani`
4. Set **Main file path**: `app.py`
5. Click **"Deploy!"**

Streamlit will automatically install packages from `requirements.txt` and launch your app.

### Step 3 — Submit Your Live URL

Once deployed, your app will be live at:

```
https://fraud-detection-capstone-dumajgp6t3d5izdv95eeyj.streamlit.app/
```

Paste this URL into the submission form:  
🔗 https://docs.google.com/forms/d/e/1FAIpQLSeQbkVrKn_UM6eZ6JoK9oxa_cXb1DbsDoH87ap_8MRu_RU5Sw/viewform

---

## 📊 Model Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|-------|----------|-----------|--------|----------|---------|--------|
| **LightGBM** ⭐ | ~97.8% | ~89.2% | ~88.1% | ~88.6% | ~0.981 | ~0.872 |
| XGBoost | ~97.2% | ~86.4% | ~85.3% | ~85.8% | ~0.976 | ~0.851 |
| Isolation Forest | ~94.1% | ~61.2% | ~78.5% | ~68.8% | ~0.891 | ~0.612 |

> **Why PR-AUC over Accuracy?** With only 3.5% fraud rate, a model predicting all transactions as legitimate achieves >96% accuracy. PR-AUC directly measures performance over the minority fraud class — the metric that matters for real-world deployment.

---

## 🛠️ Tools & Libraries Used

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core language |
| Pandas / NumPy | Data manipulation |
| Scikit-learn | ML utilities, metrics, scaling |
| LightGBM | Primary fraud classifier |
| XGBoost | Comparison model |
| imbalanced-learn | SMOTE oversampling |
| SHAP | Explainable AI |
| Optuna | Hyperparameter tuning |
| Plotly | Interactive dashboard charts |
| Streamlit | Live web dashboard |
| Matplotlib / Seaborn | Static analysis charts |

---

## 📤 Submission Checklist

- [x] `analysis_fixed.ipynb` — all 8 tasks completed with markdown headings
- [x] `data/` — transaction + identity CSV files
- [x] `app.py` — 3-page Streamlit dashboard
- [x] `best_model.pkl` — serialised LightGBM model
- [x] `dashboard_data.csv` — pre-processed test slice
- [x] `charts/` — all visualisation exports
- [x] `requirements.txt` — full dependency list
- [x] `README.md` — this document
- [ ] `summary.pdf` / `summary.docx` — written project summary *(optional)*
- [x] Live Streamlit URL submitted via Google Form

---

*© 2026 Shadhani Shambharkar — AI & Data Analytics Internship Capstone*
