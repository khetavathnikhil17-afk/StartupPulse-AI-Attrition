"""
Training Module – StartupPulse AI: Employee Attrition Prediction.

Builds, compiles, and trains a Deep Neural Network using TensorFlow/Keras.
Includes early stopping, learning-rate scheduling, and model checkpointing.
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
_PROCESSED_DIR = _ROOT / "data" / "processed"
_MODEL_DIR = _ROOT / "models" / "startuppulse_v1"
_FIGURES_DIR = _ROOT / "reports" / "figures"
_RESULTS_DIR = _ROOT / "reports" / "results"

# Training hyper-parameters
_EPOCHS = 100
_BATCH_SIZE = 32
_LEARNING_RATE = 0.001
_PATIENCE_ES = 10        # EarlyStopping patience
_PATIENCE_LR = 5         # ReduceLROnPlateau patience
_MIN_LR = 1e-6           # Minimum learning rate


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the preprocessed train / validation / test CSV files.

    Returns:
        (train_df, val_df, test_df)

    Raises:
        FileNotFoundError: If any of the CSV files are missing.
    """
    paths = {
        "train": _PROCESSED_DIR / "train.csv",
        "validation": _PROCESSED_DIR / "validation.csv",
        "test": _PROCESSED_DIR / "test.csv",
    }

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")

    train_df = pd.read_csv(paths["train"])
    val_df = pd.read_csv(paths["validation"])
    test_df = pd.read_csv(paths["test"])

    logger.info(
        "Loaded splits — train: %s | val: %s | test: %s",
        train_df.shape, val_df.shape, test_df.shape,
    )
    return train_df, val_df, test_df


def split_xy(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Separate features (X) and target (y) from a DataFrame.

    Args:
        df: A DataFrame containing an 'Attrition' column.

    Returns:
        (X, y) as NumPy arrays.
    """
    X = df.drop(columns=["Attrition"]).values.astype(np.float32)
    y = df["Attrition"].values.astype(np.float32)
    return X, y


# ---------------------------------------------------------------------------
# Class-Weight Computation
# ---------------------------------------------------------------------------
def compute_class_weights(y: np.ndarray) -> dict:
    """Compute balanced class weights for imbalanced targets.

    Args:
        y: Binary target array (0 or 1).

    Returns:
        Dict {0: weight_0, 1: weight_1} inversely proportional to frequency.
    """
    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())
    total = n_neg + n_pos

    weight_neg = total / (2.0 * n_neg)
    weight_pos = total / (2.0 * n_pos)

    weights = {0: weight_neg, 1: weight_pos}
    logger.info(
        "Class weights — 0: %.3f  |  1: %.3f  (neg=%d, pos=%d)",
        weight_neg, weight_pos, n_neg, n_pos,
    )
    return weights


# ---------------------------------------------------------------------------
# Model Building
# ---------------------------------------------------------------------------
def build_model(input_dim: int) -> keras.Model:
    """Construct the Deep Neural Network architecture.

    Architecture:
        Input → Dense(256, ReLU) → BN → Dropout(0.3)
              → Dense(128, ReLU) → BN → Dropout(0.3)
              → Dense(64, ReLU)  → BN → Dropout(0.2)
              → Dense(32, ReLU)
              → Dense(1, Sigmoid)

    Args:
        input_dim: Number of input features.

    Returns:
        A compiled Keras Model.
    """
    logger.info("Building DNN — input_dim=%d", input_dim)

    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            # Block 1
            layers.Dense(256, activation="relu", kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            # Block 2
            layers.Dense(128, activation="relu", kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            # Block 3
            layers.Dense(64, activation="relu", kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            # Block 4
            layers.Dense(32, activation="relu", kernel_initializer="he_normal"),
            # Output
            layers.Dense(1, activation="sigmoid"),
        ],
        name="AttritionDNN",
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=_LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )

    model.summary(print_fn=logger.info)
    return model


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def build_callbacks(model_path: Path) -> list:
    """Create the training callback list.

    Callbacks:
        - EarlyStopping: halt when val_loss stops improving.
        - ReduceLROnPlateau: reduce LR on plateau.
        - ModelCheckpoint: save best model by val_loss.

    Args:
        model_path: Filesystem path for the saved .keras model.

    Returns:
        List of Keras Callback instances.
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)

    cbs = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=_PATIENCE_ES,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=_PATIENCE_LR,
            min_lr=_MIN_LR,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    logger.info("Callbacks configured — EarlyStopping(patience=%d), ReduceLROnPlateau(patience=%d)",
                _PATIENCE_ES, _PATIENCE_LR)
    return cbs


# ---------------------------------------------------------------------------
# Training History Plots
# ---------------------------------------------------------------------------
def plot_training_history(history: keras.callbacks.History) -> None:
    """Generate and save accuracy and loss curves as interactive HTML.

    Saves:
        reports/figures/training_accuracy_curve.html
        reports/figures/training_loss_curve.html

    Args:
        history: Keras training History object.
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
    _save_fig(fig_acc, "training_accuracy_curve")

    # --- Loss Curve ---
    fig_loss = go.Figure()
    fig_loss.add_trace(
        go.Scatter(
            x=epochs_range, y=hist["loss"],
            mode="lines", name="Train Loss",
            line=dict(color="#636EFA", width=2),
        )
    )
    fig_loss.add_trace(
        go.Scatter(
            x=epochs_range, y=hist["val_loss"],
            mode="lines", name="Val Loss",
            line=dict(color="#EF553B", width=2, dash="dash"),
        )
    )

    # Mark the best epoch
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
    _save_fig(fig_loss, "training_loss_curve")

    logger.info("Training history plots saved.")


def _save_fig(fig: go.Figure, name: str) -> None:
    """Persist a Plotly figure as an HTML file."""
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = _FIGURES_DIR / f"{name}.html"
    fig.write_html(str(path), include_plotlyjs="cdn")
    logger.info("Saved figure: %s", path.name)


# ---------------------------------------------------------------------------
# Training Orchestrator
# ---------------------------------------------------------------------------
def train() -> Tuple[keras.Model, keras.callbacks.History]:
    """Execute the full training pipeline.

    Steps:
        1. Load preprocessed data splits
        2. Separate features / target
        3. Compute class weights
        4. Build the DNN
        5. Train with callbacks
        6. Save training history plots

    Returns:
        (trained_model, training_history)
    """
    logger.info("=" * 60)
    logger.info("STARTING TRAINING PIPELINE")
    logger.info("=" * 60)

    # 1. Load data
    train_df, val_df, _ = load_splits()

    X_train, y_train = split_xy(train_df)
    X_val, y_val = split_xy(val_df)

    input_dim = X_train.shape[1]
    logger.info("Input dimension: %d", input_dim)

    # 2. Class weights
    class_weights = compute_class_weights(y_train)

    # 3. Build model
    model_path = _MODEL_DIR / "attrition_model.keras"
    model = build_model(input_dim)

    # 4. Callbacks
    cbs = build_callbacks(model_path)

    # 5. Train
    logger.info("Training for up to %d epochs (batch_size=%d) …", _EPOCHS, _BATCH_SIZE)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=_EPOCHS,
        batch_size=_BATCH_SIZE,
        class_weight=class_weights,
        callbacks=cbs,
        verbose=1,
    )

    # 6. Plots
    plot_training_history(history)

    # 7. Final summary
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    best_val_loss = min(history.history["val_loss"])
    best_val_acc = max(history.history["val_accuracy"])

    logger.info("Training complete.")
    logger.info("  Best epoch       : %d", best_epoch)
    logger.info("  Best val_loss    : %.4f", best_val_loss)
    logger.info("  Best val_accuracy: %.4f", best_val_acc)
    logger.info("  Model saved to   : %s", model_path)

    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 60)

    return model, history


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    train()
