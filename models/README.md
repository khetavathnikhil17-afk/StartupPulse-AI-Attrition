# Model Artifacts

This directory contains the trained model files for StartupPulse AI.

## Important

**Model files are not included in version control.** The `.keras` and `.pkl` files are too large for Git and must be generated locally by training the model.

## Directory Structure

```
models/
└── startuppulse_v1/
    ├── attrition_model.keras      # Trained DNN model weights (~688 KB)
    ├── scaler.pkl                  # Fitted StandardScaler (feature normalization)
    └── label_encoders.pkl          # Fitted LabelEncoders (categorical encoding)
```

## Generating Model Files

### Prerequisites

1. Python 3.9+ with dependencies installed:
   ```bash
   pip install -r requirements.txt
   ```

2. The raw dataset must be present at `data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv`.

   Download from: [Kaggle - IBM HR Analytics Employee Attrition Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

### Step 1: Preprocess Data

```bash
python -m src.data.preprocessing
```

This generates:
- `data/processed/train.csv` (1,028 samples)
- `data/processed/validation.csv` (221 samples)
- `data/processed/test.csv` (221 samples)
- `models/startuppulse_v1/scaler.pkl`
- `models/startuppulse_v1/label_encoders.pkl`

### Step 2: Train the Model

```bash
python -m src.model.train
```

This generates:
- `models/startuppulse_v1/attrition_model.keras`
- `reports/figures/training_accuracy_curve.html`
- `reports/figures/training_loss_curve.html`

Training runs for up to 100 epochs with early stopping (patience=10). Expected runtime: 2–5 minutes depending on hardware.

### Step 3: Evaluate (Optional)

```bash
python -m src.model.evaluate
```

Generates:
- `reports/results/metrics.json`
- `reports/results/evaluation_report.md`
- `reports/figures/confusion_matrix.html`
- `reports/figures/roc_curve.html`

### Step 4: Generate SHAP Explanations (Optional)

```bash
python -m src.explainability.shap_explainer
```

Generates SHAP plots in `reports/figures/`:
- `shap_summary_plot.html`
- `shap_global_feature_importance.html`
- `shap_waterfall_sample_0.html`
- `shap_force_plot_sample_0.html`
- `shap_local_feature_importance_sample_0.html`

## Model Details

| Property | Value |
|---|---|
| Architecture | 5-layer DNN (256→128→64→32→1) |
| Framework | TensorFlow 2.x / Keras |
| Input features | 30 |
| Output | Binary (attrition probability) |
| Total parameters | 52,993 |

### Performance

| Metric | Score |
|---|---|
| Accuracy | 86.9% |
| Precision | 65.2% |
| Recall | 41.7% |
| F1-Score | 50.9% |
| ROC-AUC | 76.9% |
| Optimal Threshold | 0.888 |

## File Descriptions

### `attrition_model.keras`

The trained Keras model saved via `ModelCheckpoint`. Contains the full DNN architecture and learned weights. Loaded by the dashboard and prediction scripts.

### `scaler.pkl`

A `sklearn.preprocessing.StandardScaler` fitted on the training set. Used to normalize numerical features to zero mean and unit variance. Must be applied consistently at training and inference time.

### `label_encoders.pkl`

A dictionary of `sklearn.preprocessing.LabelEncoder` instances, one per categorical feature:
- BusinessTravel
- Department
- EducationField
- Gender
- JobRole
- MaritalStatus
- OverTime

Used to transform categorical string labels to integer encodings. Must be applied consistently at training and inference time.
