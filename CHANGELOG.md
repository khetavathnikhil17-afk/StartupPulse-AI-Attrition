# Changelog

All notable changes to StartupPulse AI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-07-21

### Added

#### Deep Neural Network Model
- 5-layer DNN architecture (256 → 128 → 64 → 32 → 1) built with TensorFlow/Keras
- Batch Normalization and Dropout regularization (0.2–0.3) across hidden layers
- He normal kernel initialization for stable gradient flow
- Binary cross-entropy loss with Adam optimizer (lr=0.001)
- Class-weight balancing to handle 84:16 attrition imbalance ({0: 0.6, 1: 3.6})
- EarlyStopping (patience=10) and ReduceLROnPlateau (factor=0.5, patience=5) callbacks
- Model checkpointing to `models/startuppulse_v1/attrition_model.keras`
- Achieves 70.6% accuracy, 55.6% recall, 68.1% ROC-AUC on held-out test set

#### Data Pipeline
- Automated preprocessing of IBM HR Analytics Employee Attrition Dataset (1,470 samples × 35 features)
- Label encoding for 8 categorical features: BusinessTravel, Department, EducationField, Gender, JobRole, MaritalStatus, OverTime
- StandardScaler normalization for 23 numerical features
- Train/validation/test split (70/15/15) with stratification
- Drops 4 non-predictive columns: EmployeeCount, EmployeeNumber, Over18, StandardHours
- Generates 30-feature clean dataset ready for model training

#### SHAP Explainability
- DeepExplainer integration for neural network interpretability
- Global feature importance rankings (top 10: OverTime, MonthlyIncome, Age, YearsWithCurrManager, DistanceFromHome)
- Local waterfall plots for individual employee prediction breakdowns
- Force plots showing positive/negative contribution drivers
- Summary plots with beeswarm visualization across all test samples
- Dependence plots for feature interaction analysis

#### Streamlit Dashboard (6 Pages)
- **Home**: Animated hero banner, 5 KPI cards, workflow visualization, quick navigation
- **Predict Attrition**: 30-field employee input form, real-time DNN prediction, confidence scoring, traffic-light risk display (Green/Red), top contributing factors per prediction
- **Analytics Dashboard**: 9 interactive Plotly visualizations — age distribution, monthly income box plots, department × attrition heatmap, correlation matrix, overtime analysis, job satisfaction breakdowns, environment satisfaction, tenure analysis, performance rating vs attrition
- **Explainable AI**: 5 SHAP visualization types — global summary, local waterfall, force plot, feature importance bar chart, dependence plots
- **Reports**: Animated metric cards, interactive confusion matrix, training loss/accuracy curves, full evaluation report rendering
- **About**: Project description, tech stack grid, dataset information, version details

#### Dashboard UI/UX
- Premium dark theme with glassmorphism design language
- Custom CSS (17KB) with 14 CSS animations
- JetBrains Mono (code) and Inter (UI) typography
- Responsive sidebar navigation with brand identity
- Consistent visual hierarchy across all pages

#### EDA & Visualization
- 12 interactive Plotly visualizations for exploratory data analysis
- Distribution analysis: attrition, gender, department, job role, age, monthly income
- Relationship analysis: overtime vs attrition, job satisfaction, environment satisfaction, income, tenure, performance rating
- Correlation heatmap across all 30 features
- Outlier detection and class imbalance identification

#### Project Infrastructure
- Modular source code structure: `src/data`, `src/model`, `src/visualization`, `src/explainability`
- Centralized configuration module (`src/utils/config.py`) for paths, features, hyperparameters
- Structured logging across all modules
- Type hints and PEP 8 compliance throughout
- `requirements.txt` with pinned version ranges
- `.gitignore` for Python, virtual environments, IDE files, and model artifacts
- MIT License

### Model Artifacts
- `models/startuppulse_v1/attrition_model.keras` — trained DNN weights (~688 KB)
- `models/startuppulse_v1/scaler.pkl` — fitted StandardScaler for feature normalization
- `models/startuppulse_v1/label_encoders.pkl` — fitted LabelEncoders for categorical features

### Reports
- `reports/results/metrics.json` — test set performance metrics (JSON)
- `reports/results/evaluation_report.md` — full evaluation report (Markdown)
- `reports/figures/` — all generated plots as interactive HTML files
- `reports/eda_summary_report.md` — EDA summary with key findings
