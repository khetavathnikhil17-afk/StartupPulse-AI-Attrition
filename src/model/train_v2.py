"""
Training Pipeline v2 – StartupPulse AI.

Improvements over v1:
  - SMOTE oversampling for class imbalance
  - Threshold tuning on validation set
  - Better recall/precision tradeoff
"""

import logging
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tensorflow import keras
from tensorflow.keras import callbacks, layers, optimizers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
_PROCESSED_DIR = _ROOT / "data" / "processed"
_MODEL_DIR = _ROOT / "models" / "startuppulse_v1"
_FIGURES_DIR = _ROOT / "reports" / "figures"
_RESULTS_DIR = _ROOT / "reports" / "results"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_splits():
    train_df = pd.read_csv(_PROCESSED_DIR / "train.csv")
    val_df = pd.read_csv(_PROCESSED_DIR / "validation.csv")
    test_df = pd.read_csv(_PROCESSED_DIR / "test.csv")
    return train_df, val_df, test_df


def split_xy(df):
    X = df.drop(columns=["Attrition"]).values.astype(np.float32)
    y = df["Attrition"].values.astype(np.float32)
    return X, y


# ---------------------------------------------------------------------------
# SMOTE
# ---------------------------------------------------------------------------
def apply_smote(X_train, y_train, sampling_strategy=0.8, random_state=42):
    """Apply SMOTE to oversample the minority class.

    Args:
        X_train: Training features.
        y_train: Training labels.
        sampling_strategy: Target ratio of minority/majority (0.8 = 80% of majority).
        random_state: Random seed.

    Returns:
        (X_resampled, y_resampled)
    """
    n_before = len(y_train)
    n_pos_before = int(y_train.sum())

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        k_neighbors=5,
    )
    X_res, y_res = smote.fit_resample(X_train, y_train)

    n_pos_after = int(y_res.sum())
    logger.info(
        "SMOTE: %d -> %d samples (positive: %d -> %d)",
        n_before, len(y_res), n_pos_before, n_pos_after,
    )
    return X_res, y_res


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(input_dim):
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(256, activation="relu", kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu", kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu", kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu", kernel_initializer="he_normal"),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="AttritionDNN_v2",
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


# ---------------------------------------------------------------------------
# Threshold Tuning
# ---------------------------------------------------------------------------
def find_optimal_threshold(y_true, y_pred_proba, beta=1.0):
    """Find the threshold that maximizes F-beta score on validation set.

    Args:
        y_true: Ground-truth labels.
        y_pred_proba: Predicted probabilities.
        beta: Beta value for F-beta (beta>1 favors recall).

    Returns:
        (optimal_threshold, best_f1, results_df)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)

    f1_scores = []
    for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
        if p + r == 0:
            f1_scores.append(0.0)
        else:
            f_beta = (1 + beta**2) * (p * r) / (beta**2 * p + r)
            f1_scores.append(f_beta)

    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])

    results_df = pd.DataFrame({
        "threshold": thresholds,
        "precision": precisions[:-1],
        "recall": recalls[:-1],
        "f1_score": f1_scores,
    })

    logger.info("Optimal threshold: %.4f (F1=%.4f)", best_threshold, best_f1)
    return best_threshold, best_f1, results_df


# ---------------------------------------------------------------------------
# Training Orchestrator
# ---------------------------------------------------------------------------
def train_v2():
    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE v2 (SMOTE + Threshold Tuning)")
    logger.info("=" * 60)

    # 1. Load
    train_df, val_df, test_df = load_splits()
    X_train, y_train = split_xy(train_df)
    X_val, y_val = split_xy(val_df)
    X_test, y_test = split_xy(test_df)

    input_dim = X_train.shape[1]
    logger.info("Input dim: %d | Train: %d | Val: %d | Test: %d",
                input_dim, len(y_train), len(y_val), len(y_test))

    # 2. SMOTE
    X_train_sm, y_train_sm = apply_smote(X_train, y_train, sampling_strategy=0.8)

    # 3. Build
    model = build_model(input_dim)
    model_path = _MODEL_DIR / "attrition_model.keras"
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)

    cbs = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=10,
            restore_best_weights=True, verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=5, min_lr=1e-6, verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=str(model_path), monitor="val_loss",
            save_best_only=True, verbose=1,
        ),
    ]

    # 4. Train
    logger.info("Training with SMOTE data …")
    history = model.fit(
        X_train_sm, y_train_sm,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=cbs,
        verbose=1,
    )

    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    best_val_loss = min(history.history["val_loss"])
    best_val_acc = max(history.history["val_accuracy"])
    logger.info("Best epoch: %d | val_loss=%.4f | val_acc=%.4f",
                best_epoch, best_val_loss, best_val_acc)

    # 5. Threshold tuning on validation set
    y_val_proba = model.predict(X_val, verbose=0).flatten()
    optimal_threshold, best_f1, threshold_df = find_optimal_threshold(y_val, y_val_proba, beta=1.0)

    # Save threshold analysis
    threshold_df.to_csv(_RESULTS_DIR / "threshold_analysis.csv", index=False)
    logger.info("Threshold analysis saved.")

    # 6. Evaluate on test set with optimal threshold
    y_test_proba = model.predict(X_test, verbose=0).flatten()
    y_test_pred = (y_test_proba >= optimal_threshold).astype(int)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_test_pred), 4),
        "precision": round(precision_score(y_test, y_test_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_test_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_test_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_test_proba), 4),
        "optimal_threshold": round(optimal_threshold, 4),
    }

    logger.info("Test metrics (threshold=%.4f):", optimal_threshold)
    for k, v in metrics.items():
        logger.info("  %-20s: %s", k, v)

    # 7. Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    logger.info("Confusion matrix: TN=%d FP=%d FN=%d TP=%d", tn, fp, fn, tp)

    # 8. Classification report
    logger.info("\n%s", classification_report(y_test, y_test_pred, target_names=["No", "Yes"]))

    # 9. Save
    import json
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", _RESULTS_DIR / "metrics.json")

    # 10. Save training curves
    _save_training_curves(history)

    logger.info("=" * 60)
    logger.info("TRAINING v2 COMPLETE")
    logger.info("=" * 60)

    return model, history, metrics


def _save_training_curves(history):
    import plotly.graph_objects as go

    hist = history.history
    epochs = list(range(1, len(hist["loss"]) + 1))

    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(x=epochs, y=hist["accuracy"], mode="lines", name="Train Acc", line=dict(color="#636EFA", width=2)))
    fig_acc.add_trace(go.Scatter(x=epochs, y=hist["val_accuracy"], mode="lines", name="Val Acc", line=dict(color="#EF553B", width=2, dash="dash")))
    fig_acc.update_layout(title="Training & Validation Accuracy", xaxis_title="Epoch", yaxis_title="Accuracy", template="plotly_white", height=450, width=800, legend=dict(x=0.02, y=0.98))
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_acc.write_html(str(_FIGURES_DIR / "training_accuracy_curve.html"), include_plotlyjs="cdn")

    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(x=epochs, y=hist["loss"], mode="lines", name="Train Loss", line=dict(color="#636EFA", width=2)))
    fig_loss.add_trace(go.Scatter(x=epochs, y=hist["val_loss"], mode="lines", name="Val Loss", line=dict(color="#EF553B", width=2, dash="dash")))
    best_epoch = int(np.argmin(hist["val_loss"])) + 1
    fig_loss.add_vline(x=best_epoch, line_dash="dot", line_color="green", annotation_text=f"Best: {best_epoch}")
    fig_loss.update_layout(title="Training & Validation Loss", xaxis_title="Epoch", yaxis_title="Loss", template="plotly_white", height=450, width=800, legend=dict(x=0.02, y=0.98))
    fig_loss.write_html(str(_FIGURES_DIR / "training_loss_curve.html"), include_plotlyjs="cdn")
    logger.info("Training curves saved.")


if __name__ == "__main__":
    train_v2()
