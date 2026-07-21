<div align="center">

# 🚀 StartupPulse AI

### **Employee Attrition Prediction & Explainable AI Dashboard**

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.60-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-4CAF50?style=for-the-badge)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)

---

**An end-to-end AI/ML pipeline that predicts employee attrition using a deep neural network with 56% recall, 68% ROC-AUC, and a fully interactive, glassmorphism-styled dashboard powered by SHAP explainability.**

[![Dashboard](https://img.shields.io/badge/Run_Dashboard-FF6B6B?style=for-the-badge)](#-quick-start)

---

</div>

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Solution Overview](#-solution-overview)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Key Features](#-key-features)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Dashboard Pages](#-dashboard-pages)
- [Future Scope](#-future-scope)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Problem Statement

> Employee attrition (voluntary turnover) costs U.S. companies an estimated **$1 trillion annually** (Gallup, 2023). For startups and scale-ups, losing a single key engineer can delay product launches by months. Traditional HR analytics rely on lagging indicators — exit interviews, satisfaction surveys — which fail to flag at-risk employees in time.

**StartupPulse AI** solves this by building an **explainable deep learning system** that:
- Predicts which employees are likely to leave **before** they resign
- Explains **why** each prediction was made (SHAP values)
- Presents insights through a **premium, interactive dashboard** designed for HR leaders

---

## 💡 Solution Overview

```
Raw IBM HR Dataset (1,470 employees × 35 features)
         │
         ▼
┌─────────────────────────┐
│   DATA PREPROCESSING    │  • Drop 4 irrelevant columns
│   (Label Encoding +     │  • Label encode 8 categorical features
│    Standard Scaling)    │  • Train/Val/Test split (70/15/15)
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│   DEEP NEURAL NETWORK   │  • 5-layer DNN (256→128→64→32→1)
│   (TensorFlow/Keras)    │  • BatchNorm + Dropout regularization
│   + Class Weights       │  • EarlyStopping + LR scheduling
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│   MODEL EVALUATION      │  • Accuracy, Precision, Recall, F1, ROC-AUC
│                         │  • Confusion Matrix, Classification Report
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│   SHAP EXPLAINABILITY   │  • DeepExplainer on test samples
│                         │  • Global + Local + Waterfall plots
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│   STREAMLIT DASHBOARD   │  • 6 premium pages with glassmorphism UI
│   (Interactive Web App) │  • Real-time prediction interface
│                         │  • 9+ interactive Plotly visualizations
└─────────────────────────┘
```

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        STARTUPPULSE AI ARCHITECTURE                     │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │  RAW DATA    │───▶│ PREPROCESSOR │───▶│  DNN MODEL   │               │
│  │  (1470×35)   │    │  Scaler+Enc  │    │  (5 layers)  │               │
│  └──────────────┘    └──────────────┘    └──────┬───────┘               │
│                                                  │                       │
│                          ┌───────────────────────┼───────────┐          │
│                          ▼                       ▼           ▼          │
│                  ┌──────────────┐    ┌──────────────┐ ┌──────────┐     │
│                  │   EVALUATION │    │ SHAP ENGINE  │ │PREDICTION│     │
│                  │  Metrics+CM  │    │ DeepExplainer│ │ Batch/   │     │
│                  └──────────────┘    └──────────────┘ │ Single   │     │
│                          │               │            └──────────┘     │
│                          ▼               ▼                            │
│                  ┌─────────────────────────────────────┐               │
│                  │       STREAMLIT DASHBOARD           │               │
│                  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌──────┐  │               │
│                  │  │Home │ │Pred.│ │Expl.│ │Anlyt.│  │               │
│                  │  └─────┘ └─────┘ └─────┘ └──────┘  │               │
│                  └─────────────────────────────────────┘               │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.9+ |
| **ML/DL** | TensorFlow 2.x / Keras, Scikit-learn, NumPy, Pandas |
| **Explainability** | SHAP (DeepExplainer) |
| **Visualization** | Plotly 6.x (interactive), Matplotlib (static) |
| **Frontend** | Streamlit 1.60+, custom CSS (glassmorphism) |
| **Fonts** | JetBrains Mono (code), Inter (UI) |
| **Version Control** | Git, GitHub |
| **IDE** | VS Code |

---

## 📂 Project Structure

```
StartupPulse-AI-Attrition/
│
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 .gitignore
├── 📄 eda_summary_report.md
│
├── 📁 data/
│   ├── 📁 raw/                          # Original IBM HR dataset
│   │   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│   └── 📁 processed/                    # Preprocessed splits
│       ├── train.csv                    # 1,028 samples (70%)
│       ├── validation.csv               # 221 samples (15%)
│       └── test.csv                     # 221 samples (15%)
│
├── 📁 models/
│   └── 📁 startuppulse_v1/
│       ├── attrition_model.keras        # Trained DNN model (688 KB)
│       ├── scaler.pkl                   # Fitted StandardScaler
│       └── label_encoders.pkl           # Fitted LabelEncoders
│
├── 📁 src/                              # Source code modules
│   ├── 📁 data/
│   │   ├── check_dataset.py             # Dataset exploration (14 functions)
│   │   └── preprocessing.py             # Full pipeline (10 functions)
│   ├── 📁 model/
│   │   ├── train.py                     # DNN build + train + callbacks
│   │   ├── evaluate.py                  # Metrics + confusion matrix + report
│   │   └── predict.py                   # Prediction + preprocessing
│   ├── 📁 visualization/
│   │   └── eda.py                       # 12 Plotly EDA visualizations
│   └── 📁 explainability/
│       └── shap_explainer.py            # 5 SHAP plot types
│
├── 📁 dashboard/
│   ├── app.py                           # Main Streamlit entry point
│   ├── 📁 components/
│   │   └── reusable_widgets.py          # Premium CSS (17KB, 14 animations)
│   └── 📁 pages/
│       ├── home.py                      # Hero + KPIs + workflow
│       ├── predict.py                   # 30-field prediction form
│       ├── analytics.py                 # 9 interactive charts
│       ├── explainability.py            # 5 SHAP visualizations
│       ├── reports.py                   # Metrics + training curves
│       └── about.py                     # Tech stack + team info
│
├── 📁 reports/
│   ├── 📁 figures/                      # All generated plots (HTML)
│   │   ├── 1_distribution_*.html        # EDA plots
│   │   ├── confusion_matrix.html        # Confusion matrix
│   │   ├── training_curves.html         # Loss/Accuracy curves
│   │   └── shap_*.html                  # SHAP explanation plots
│   └── 📁 results/
│       ├── evaluation_report.md         # Full evaluation report
│       └── metrics.json                 # Test set metrics
```

---

## ✨ Key Features

| Category | Feature |
|---|---|
| 🧠 **Deep Learning** | 5-layer DNN with BatchNorm + Dropout for regularization |
| ⚖️ **Class Imbalance** | Handles 84:16 class split via `class_weight` parameter |
| 🔍 **Explainability** | 5 SHAP plot types — global, local, waterfall, force, summary |
| 📊 **EDA** | 12 interactive Plotly visualizations with outlier/imbalance detection |
| 🎨 **Dashboard** | Premium dark theme with glassmorphism, 14 CSS animations |
| 🔮 **Prediction** | Real-time single/batch prediction with employee form input |
| 📈 **Analytics** | 9 interactive charts including correlation heatmap |
| 📋 **Reports** | One-click evaluation metrics, confusion matrix, training curves |
| 🏗️ **Modular** | Clean separation — data, model, explainability, dashboard |
| 📝 **Production-Ready** | Logging, type hints, exception handling, PEP8 compliant |

---

## 📊 Dataset

**IBM HR Analytics Employee Attrition Dataset**

| Property | Value |
|---|---|
| **Source** | [Kaggle - IBM HR Analytics](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) |
| **Samples** | 1,470 employees |
| **Features** | 35 columns (30 after preprocessing) |
| **Target** | `Attrition` (Yes/No → 1/0) |
| **Class Distribution** | No: 83.9% / Yes: 16.1% (imbalanced) |
| **Categorical Features** | BusinessTravel, Department, EducationField, Gender, JobRole, MaritalStatus, OverTime |
| **Numerical Features** | Age, DailyRate, DistanceFromHome, Education, EnvironmentSatisfaction, HourlyRate, JobInvolvement, JobLevel, JobSatisfaction, MonthlyIncome, MonthlyRate, NumCompaniesWorked, PercentSalaryHike, PerformanceRating, RelationshipSatisfaction, StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager |

**Dropped Columns:** EmployeeCount, EmployeeNumber, Over18, StandardHours (no predictive value)

---

## 🧠 Model Architecture

```
Layer (type)                 Output Shape              Param #
═══════════════════════════════════════════════════════════════
input (InputLayer)           (None, 30)                0
dense (Dense, ReLU)          (None, 256)               7,936
batch_normalization (BN)     (None, 256)               1,024
dropout (Dropout 0.3)        (None, 256)               0
dense_1 (Dense, ReLU)        (None, 128)               32,896
batch_normalization_1 (BN)   (None, 128)               512
dropout_1 (Dropout 0.3)      (None, 128)               0
dense_2 (Dense, ReLU)        (None, 64)                8,256
batch_normalization_2 (BN)   (None, 64)                256
dropout_2 (Dropout 0.2)      (None, 64)                0
dense_3 (Dense, ReLU)        (None, 32)                2,080
output (Dense, Sigmoid)      (None, 1)                 33
═══════════════════════════════════════════════════════════════
Total params: 52,993
Trainable params: 52,225
Non-trainable params: 768
```

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam (lr=0.001) |
| Loss Function | Binary Cross-Entropy |
| Max Epochs | 100 |
| Batch Size | 32 |
| Early Stopping | patience=10, monitor=val_loss |
| LR Scheduler | ReduceLROnPlateau, factor=0.5, patience=5 |
| Class Weights | {0: 0.6, 1: 3.6} (inverse frequency) |

---

## 📈 Results

| Metric | Score |
|---|---|
| **Accuracy** | **70.6%** |
| **Precision** | 35.4% |
| **Recall** | **55.6%** |
| **F1-Score** | 45.5% |
| **ROC-AUC** | **68.1%** |

> The model achieves **recall (55.6%)** — critical for attrition prediction, as missing an at-risk employee (false negative) is costlier than a false alarm.

### Confusion Matrix

```
                Predicted
              No     Yes
Actual  No  [143     42]
        Yes [ 13     23]
```

### Top 10 Most Important Features (SHAP)

| Rank | Feature | Mean |SHAP Value |
|---|---|---|
| 1 | **OverTime** | 0.123 |
| 2 | **MonthlyIncome** | 0.109 |
| 3 | **Age** | 0.074 |
| 4 | **YearsWithCurrManager** | 0.066 |
| 5 | **DistanceFromHome** | 0.062 |
| 6 | **TotalWorkingYears** | 0.055 |
| 7 | **StockOptionLevel** | 0.051 |
| 8 | **YearsAtCompany** | 0.048 |
| 9 | **JobRole** | 0.043 |
| 10 | **JobSatisfaction** | 0.039 |

---

## 🛠️ Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- 4 GB+ RAM recommended

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/StartupPulse-AI-Attrition.git
cd StartupPulse-AI-Attrition

# 2. Create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset from Kaggle and place in data/raw/
#    https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

# 5. Run preprocessing
python -m src.data.preprocessing

# 6. Train the model
python -m src.model.train

# 7. Evaluate the model
python -m src.model.evaluate

# 8. Generate SHAP explanations
python -m src.explainability.shap_explainer
```

---

## 🚀 Quick Start

### Launch the Dashboard

```bash
cd StartupPulse-AI-Attrition
streamlit run dashboard/app.py
```

The dashboard will open at **http://localhost:8501** with 6 premium pages:

| Page | Description |
|---|---|
| 🏠 **Home** | Hero banner, KPIs, workflow overview |
| 🔮 **Predict** | 30-field employee form → real-time prediction |
| 📊 **Analytics** | 9 interactive charts + correlation heatmap |
| 🔍 **Explainability** | 5 SHAP plots + feature importance bars |
| 📈 **Reports** | Metrics, confusion matrix, training curves |
| ℹ️ **About** | Project description, tech stack, dataset info |

### Run Individual Scripts

```bash
# Explore dataset
python -m src.data.check_dataset

# Generate EDA plots
python -m src.visualization.eda

# Batch prediction
python -m src.model.predict --input data/processed/test.csv --output predictions.csv

# Single prediction
python -m src.model.predict --employee '{"Age":35,"MonthlyIncome":5000,...}'
```

---

## 🖥️ Dashboard Pages

### 🏠 Home
- Animated hero banner with glassmorphism
- 5 KPI cards (model accuracy, dataset size, features, recall, ROC-AUC)
- Step-by-step workflow visualization
- Quick navigation cards

### 🔮 Prediction
- 30-field employee form (Age, Salary, Department, etc.)
- One-click prediction with confidence score
- Traffic-light result display (Green = Safe, Red = At Risk)
- Top contributing factors shown per prediction

### 📊 Analytics
- Age distribution by attrition status
- Monthly income comparison (box plots)
- Department × attrition heatmap
- Correlation matrix (all 30 features)
- OverTime vs attrition bar chart

### 🔍 Explainability
- Global SHAP summary plot (feature importance ranking)
- Local SHAP waterfall plot (single employee explanation)
- SHAP force plot (prediction breakdown)
- SHAP dependence plots (feature interactions)

### 📈 Reports
- 5 metric cards with animated counters
- Confusion matrix (interactive)
- Training loss/accuracy curves
- Full evaluation report (markdown rendered)

### ℹ️ About
- Project description and motivation
- Tech stack grid with icons
- Dataset details table
- Version and contact info

---

## 🔮 Future Scope

| Enhancement | Description |
|---|---|
| **Real-Time Data Integration** | Connect to live HRIS databases (Workday, BambooHR) |
| **Advanced Models** | Experiment with XGBoost, LightGBM, Transformer-based tabular models |
| **A/B Testing Framework** | Compare prediction strategies and model versions |
| **Deployment** | Docker container + AWS/GCP deployment with CI/CD |
| **API Endpoint** | FastAPI REST endpoint for production integration |
| **Slack/Teams Bot** | Alert HR teams when attrition risk exceeds threshold |
| **Time-Series Analysis** | Track attrition trends over quarters |
| **Multi-Company Model** | Train on combined datasets from multiple organizations |

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

```bash
# 1. Fork the repository
# 2. Create a feature branch
git checkout -b feature/your-feature-name

# 3. Make your changes
# 4. Commit your changes
git commit -m "Add: your feature description"

# 6. Push to the branch
git push origin feature/your-feature-name

# 7. Create a Pull Request
```

---

## 📜 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2026 StartupPulse AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

<div align="center">

### ⭐ Star this repo if you found it useful!

Built with ❤️ by **StartupPulse AI Team**

---

![Visitors](https://api.visitorbadge.io/api/visitors?path=yourusername%2FStartupPulse-AI-Attrition&countColor=%232563EB&style=for-the-badge)

</div>
