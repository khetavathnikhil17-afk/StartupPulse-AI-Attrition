"""
Evaluation Module – StartupPulse AI: Employee Attrition Prediction.

Loads the trained model, evaluates it on the held-out test set,
computes all required metrics, generates the confusion matrix
heatmap, and writes a Markdown evaluation report.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tensorflow import keras

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
_PROCESSED_DIR = _ROOT / "data" / "processed"
_MODEL_DIR = _ROOT / "models" / "startuppulse_v1"
_FIGURES_DIR = _ROOT / "reports" / "figures"
_RESULTS_DIR = _ROOT / "reports" / "results"


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_test_split() -> Tuple[np.ndarray, np.ndarray]:
    """Load the test CSV and separate features / target.

    Returns:
        (X_test, y_test) as NumPy arrays.

    Raises:
        FileNotFoundError: If test.csv is missing.
    """
    path = _PROCESSED_DIR / "test.csv"
    if not path.exists():
        raise FileNotFoundError(f"Test split not found: {path}")

    df = pd.read_csv(path)
    X = df.drop(columns=["Attrition"]).values.astype(np.float32)
    y = df["Attrition"].values.astype(np.float32)

    logger.info("Test split loaded — X: %s, y: %s", X.shape, y.shape)
    return X, y


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------
def load_model(path: Path = _MODEL_DIR / "attrition_model.keras") -> keras.Model:
    """Load the saved Keras model from disk.

    Args:
        path: Filesystem path to the .keras file.

    Returns:
        A compiled Keras Model.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    model = keras.models.load_model(path)
    logger.info("Model loaded from %s", path)
    return model


# ---------------------------------------------------------------------------
# Metric Computation
# ---------------------------------------------------------------------------
def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute classification metrics from predicted probabilities.

    Args:
        y_true: Ground-truth binary labels.
        y_pred_proba: Predicted probabilities from the sigmoid output.
        threshold: Classification threshold (default 0.5).

    Returns:
        Dict with keys: accuracy, precision, recall, f1, roc_auc.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_pred_proba), 4),
    }

    logger.info("Evaluation metrics:")
    for k, v in metrics.items():
        logger.info("  %-12s: %.4f", k, v)

    return metrics


# ---------------------------------------------------------------------------
# Confusion Matrix Heatmap
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save: bool = True,
) -> go.Figure:
    """Generate an interactive Plotly confusion matrix heatmap.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Binary predictions (0 or 1).
        save: Whether to persist the figure to disk.

    Returns:
        A Plotly Figure object.
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = ["No Attrition", "Yes Attrition"]

    # Build annotated text
    text = [[f"{val}" for val in row] for row in cm]

    # Compute percentages for each cell
    cm_pct = cm.astype(float) / cm.sum() * 100
    annotations = [
        [f"{cm[i][j]}<br>({cm_pct[i][j]:.1f}%)" for j in range(2)]
        for i in range(2)
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=annotations,
            texttemplate="%{text}",
            textfont={"size": 16},
            showscale=False,
            hoverongaps=False,
            hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Confusion Matrix — Test Set",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        template="plotly_white",
        height=450,
        width=550,
        xaxis={"side": "bottom"},
    )

    if save:
        _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        out_path = _FIGURES_DIR / "confusion_matrix.html"
        fig.write_html(str(out_path), include_plotlyjs="cdn")
        logger.info("Confusion matrix saved to %s", out_path)

    return fig


# ---------------------------------------------------------------------------
# ROC Curve
# ---------------------------------------------------------------------------
def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save: bool = True,
) -> go.Figure:
    """Generate an interactive Plotly ROC curve.

    Args:
        y_true: Ground-truth binary labels.
        y_pred_proba: Predicted probabilities from the sigmoid output.
        save: Whether to persist the figure to disk.

    Returns:
        A Plotly Figure object.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)

    fig = go.Figure()

    # ROC curve
    fig.add_trace(
        go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=f"ROC Curve (AUC = {auc_score:.4f})",
            line=dict(color="#636EFA", width=3),
            fill="tozeroy",
            fillcolor="rgba(99, 110, 250, 0.1)",
        )
    )

    # Diagonal baseline
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="Random Baseline",
            line=dict(color="#EF553B", width=2, dash="dash"),
        )
    )

    # Optimal threshold point (Youden's J)
    j_scores = tpr - fpr
    optimal_idx = int(np.argmax(j_scores))
    optimal_threshold = thresholds[optimal_idx]

    fig.add_trace(
        go.Scatter(
            x=[fpr[optimal_idx]], y=[tpr[optimal_idx]],
            mode="markers+text",
            name=f"Optimal Threshold ({optimal_threshold:.3f})",
            marker=dict(size=12, color="#00CC96", symbol="star"),
            text=[f"Threshold: {optimal_threshold:.3f}"],
            textposition="top right",
            textfont=dict(size=11, color="#00CC96"),
        )
    )

    fig.update_layout(
        title="ROC Curve — Test Set",
        xaxis_title="False Positive Rate (1 - Specificity)",
        yaxis_title="True Positive Rate (Sensitivity / Recall)",
        template="plotly_white",
        height=500,
        width=650,
        legend=dict(x=0.55, y=0.05),
        xaxis=dict(range=[-0.02, 1.02]),
        yaxis=dict(range=[-0.02, 1.02]),
    )

    if save:
        _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        out_path = _FIGURES_DIR / "roc_curve.html"
        fig.write_html(str(out_path), include_plotlyjs="cdn")
        logger.info("ROC curve saved to %s", out_path)

    return fig


# ---------------------------------------------------------------------------
# Training Curves (Loss + Accuracy)
# ---------------------------------------------------------------------------
def plot_training_curves(
    history: "keras.callbacks.History",
    save: bool = True,
) -> None:
    """Generate and save training loss and accuracy curves.

    Args:
        history: Keras training History object.
        save: Whether to persist the figures to disk.
    """
    hist = history.history
    epochs_range = list(range(1, len(hist["loss"]) + 1))

    # --- Accuracy Curve ---
    fig_acc = go.Figure()
    fig_acc.add_trace(
        go.Scatter(
            x=epochs_range, y=hist["accuracy"],
            mode="lines", name="Train Accuracy",
            line=dict(color="#636EFA", width=2),
        )
    )
    if "val_accuracy" in hist:
        fig_acc.add_trace(
            go.Scatter(
                x=epochs_range, y=hist["val_accuracy"],
                mode="lines", name="Val Accuracy",
                line=dict(color="#EF553B", width=2, dash="dash"),
            )
        )
    fig_acc.update_layout(
        title="Training & Validation Accuracy",
        xaxis_title="Epoch",
        yaxis_title="Accuracy",
        template="plotly_white",
        height=450, width=800,
        legend=dict(x=0.02, y=0.98),
    )

    # --- Loss Curve ---
    fig_loss = go.Figure()
    fig_loss.add_trace(
        go.Scatter(
            x=epochs_range, y=hist["loss"],
            mode="lines", name="Train Loss",
            line=dict(color="#636EFA", width=2),
        )
    )
    if "val_loss" in hist:
        fig_loss.add_trace(
            go.Scatter(
                x=epochs_range, y=hist["val_loss"],
                mode="lines", name="Val Loss",
                line=dict(color="#EF553B", width=2, dash="dash"),
            )
        )
        best_epoch = int(np.argmin(hist["val_loss"])) + 1
        best_val_loss = min(hist["val_loss"])
        fig_loss.add_vline(
            x=best_epoch, line_dash="dot", line_color="green",
            annotation_text=f"Best epoch: {best_epoch} (loss={best_val_loss:.4f})",
        )

    fig_loss.update_layout(
        title="Training & Validation Loss",
        xaxis_title="Epoch",
        yaxis_title="Binary Crossentropy Loss",
        template="plotly_white",
        height=450, width=800,
        legend=dict(x=0.02, y=0.98),
    )

    if save:
        _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig_acc.write_html(str(_FIGURES_DIR / "training_accuracy_curve.html"), include_plotlyjs="cdn")
        fig_loss.write_html(str(_FIGURES_DIR / "training_loss_curve.html"), include_plotlyjs="cdn")
        logger.info("Training curves saved.")


# ---------------------------------------------------------------------------
# Evaluation Report (Markdown)
# ---------------------------------------------------------------------------
def generate_evaluation_report(
    metrics: Dict[str, float],
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
    report_path: Path = _RESULTS_DIR / "evaluation_report.md",
) -> None:
    """Write a Markdown evaluation report.

    Args:
        metrics: Dict returned by compute_metrics().
        y_true: Ground-truth labels.
        y_pred_proba: Predicted probabilities.
        threshold: Classification threshold.
        report_path: Output file path.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    clf_report = classification_report(
        y_true, y_pred, target_names=["No Attrition", "Yes Attrition"]
    )

    total = len(y_true)
    n_correct = int((y_true == y_pred).sum())
    n_wrong = total - n_correct

    report = f"""# StartupPulse AI – Evaluation Report

> **Model:** AttritionDNN (TensorFlow/Keras)
> **Test Set:** {total} samples
> **Threshold:** {threshold}

---

## 1. Metrics Summary

| Metric | Value |
|--------|-------|
| Accuracy | {metrics['accuracy']:.4f} |
| Precision | {metrics['precision']:.4f} |
| Recall | {metrics['recall']:.4f} |
| F1 Score | {metrics['f1_score']:.4f} |
| ROC AUC | {metrics['roc_auc']:.4f} |

---

## 2. Confusion Matrix

```
                Predicted
              No       Yes
Actual No  | {cm[0][0]:>5}  | {cm[0][1]:>5} |
Actual Yes | {cm[1][0]:>5}  | {cm[1][1]:>5} |
```

| Metric | Count |
|--------|-------|
| True Negatives (TN) | {cm[0][0]} |
| False Positives (FP) | {cm[0][1]} |
| False Negatives (FN) | {cm[1][0]} |
| True Positives (TP) | {cm[1][1]} |
| Correct Predictions | {n_correct} / {total} ({n_correct/total*100:.1f}%) |
| Incorrect Predictions | {n_wrong} / {total} ({n_wrong/total*100:.1f}%) |

---

## 3. Classification Report

```
{clf_report}
```

---

## 4. Interpretation

- **Precision ({metrics['precision']:.4f}):** Of employees predicted to leave,
  {metrics['precision']*100:.1f}% actually did. Lower precision means more
  false alarms (flagging happy employees as flight risks).

- **Recall ({metrics['recall']:.4f}):** Of employees who actually left,
  the model caught {metrics['recall']*100:.1f}%. Higher recall is critical
  in attrition prediction — missing a leaver is costlier than a false alarm.

- **F1 Score ({metrics['f1_score']:.4f}):** Harmonic mean of precision and
  recall. Balances the trade-off between the two.

- **ROC AUC ({metrics['roc_auc']:.4f}):** Probability that the model ranks
  a random positive higher than a random negative. Above 0.80 is generally
  considered good; above 0.85 is strong.

---

## 5. Business Recommendations

1. **Prioritize recall** — it is better to intervene with a happy employee
   than to lose a valuable one silently.
2. **Lower the threshold** (e.g., 0.3-0.4) if recall needs to increase at
   the cost of precision.
3. **Combine model predictions** with manager intuition for final decisions.
4. **Re-train quarterly** as workforce dynamics evolve.

---

*Report generated automatically by `src.model.evaluate`*
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    logger.info("Evaluation report saved to %s", report_path)


# ---------------------------------------------------------------------------
# Save Metrics JSON
# ---------------------------------------------------------------------------
def save_metrics_json(
    metrics: Dict[str, float],
    output_path: Path = _RESULTS_DIR / "metrics.json",
) -> None:
    """Persist evaluation metrics as a JSON file.

    Args:
        metrics: Dict returned by compute_metrics().
        output_path: Output file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics JSON saved to %s", output_path)


# ---------------------------------------------------------------------------
# Evaluation Orchestrator
# ---------------------------------------------------------------------------
def evaluate() -> Dict[str, float]:
    """Execute the full evaluation pipeline.

    Steps:
        1. Load the trained model
        2. Load the test split
        3. Predict probabilities
        4. Compute all metrics
        5. Generate confusion matrix plot
        6. Write evaluation report + metrics JSON

    Returns:
        Dict of evaluation metrics.
    """
    logger.info("=" * 60)
    logger.info("STARTING EVALUATION PIPELINE")
    logger.info("=" * 60)

    # 1. Load model
    model = load_model()

    # 2. Load test data
    X_test, y_test = load_test_split()

    # 3. Predict
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_proba >= 0.5).astype(int)

    logger.info("Predictions generated — shape: %s", y_pred_proba.shape)

    # 4. Metrics
    metrics = compute_metrics(y_test, y_pred_proba)

    # 5. Confusion matrix
    plot_confusion_matrix(y_test, y_pred, save=True)

    # 6. ROC Curve
    plot_roc_curve(y_test, y_pred_proba, save=True)

    # 7. Reports
    generate_evaluation_report(metrics, y_test, y_pred_proba)
    save_metrics_json(metrics)

    logger.info("=" * 60)
    logger.info("EVALUATION PIPELINE COMPLETE")
    logger.info("=" * 60)

    return metrics


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    evaluate()
