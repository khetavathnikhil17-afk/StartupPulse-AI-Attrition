"""
SHAP Explainability Module – StartupPulse AI: Employee Attrition Prediction.

Uses SHAP (SHapley Additive exPlanations) to explain the Keras DNN model.
Generates global and local feature importance, waterfall, force, and
summary plots — all saved as interactive HTML files.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
from tensorflow import keras

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = _ROOT / "models" / "startuppulse_v1"
_PROCESSED_DIR = _ROOT / "data" / "processed"
_FIGURES_DIR = _ROOT / "reports" / "figures"

# Canonical feature names (must match training order)
_FEATURE_NAMES: List[str] = [
    "Age", "BusinessTravel", "DailyRate", "Department", "DistanceFromHome",
    "Education", "EducationField", "EnvironmentSatisfaction", "Gender",
    "HourlyRate", "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction",
    "MaritalStatus", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked",
    "OverTime", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
]


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_background_data(
    n_samples: int = 100,
    path: Path = _PROCESSED_DIR / "train.csv",
) -> np.ndarray:
    """Load a random subsample of training data as SHAP background.

    Args:
        n_samples: Number of background samples.
        path: Path to the training CSV.

    Returns:
        NumPy array of shape (n_samples, n_features).
    """
    if not path.exists():
        raise FileNotFoundError(f"Background data not found: {path}")
    df = pd.read_csv(path)
    X = df.drop(columns=["Attrition"]).values.astype(np.float32)

    if X.shape[0] > n_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(X.shape[0], size=n_samples, replace=False)
        X = X[idx]

    logger.info("Background data loaded — shape: %s", X.shape)
    return X


def load_test_data(
    path: Path = _PROCESSED_DIR / "test.csv",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load the test split.

    Returns:
        (X_test, y_test) as float32 NumPy arrays.
    """
    if not path.exists():
        raise FileNotFoundError(f"Test data not found: {path}")
    df = pd.read_csv(path)
    X = df.drop(columns=["Attrition"]).values.astype(np.float32)
    y = df["Attrition"].values.astype(np.float32)
    logger.info("Test data loaded — X: %s, y: %s", X.shape, y.shape)
    return X, y


def load_model(
    path: Path = _MODEL_DIR / "attrition_model.keras",
) -> keras.Model:
    """Load the trained Keras model.

    Returns:
        Compiled Keras Model.
    """
    model = keras.models.load_model(path)
    logger.info("Model loaded from %s", path)
    return model


# ---------------------------------------------------------------------------
# SHAP Explainer Construction
# ---------------------------------------------------------------------------
def build_shap_explainer(
    model: keras.Model,
    background: np.ndarray,
) -> shap.DeepExplainer:
    """Construct a SHAP DeepExplainer for the Keras DNN.

    Args:
        model: Trained Keras model.
        background: Background dataset for SHAP.

    Returns:
        A fitted DeepExplainer instance.
    """
    logger.info("Building SHAP DeepExplainer …")
    explainer = shap.DeepExplainer(model, background)
    logger.info("SHAP explainer ready.")
    return explainer


def compute_shap_values(
    explainer: shap.DeepExplainer,
    X: np.ndarray,
) -> np.ndarray:
    """Compute SHAP values for the given input data.

    Args:
        explainer: Fitted DeepExplainer.
        X: Input data to explain, shape (n_samples, n_features).

    Returns:
        SHAP values array, shape (n_samples, n_features).
    """
    logger.info("Computing SHAP values for %d samples …", X.shape[0])
    raw = explainer.shap_values(X)

    # DeepExplainer may return a list of arrays for multi-output;
    # for binary classification with sigmoid output, take index 0.
    if isinstance(raw, list):
        shap_vals = raw[0]
    else:
        shap_vals = raw

    # Ensure 2-D: (n_samples, n_features)
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 0]

    logger.info("SHAP values computed — shape: %s", shap_vals.shape)
    return shap_vals


# ---------------------------------------------------------------------------
# Plot 1: Global Feature Importance (Bar Chart)
# ---------------------------------------------------------------------------
def plot_global_feature_importance(
    shap_vals: np.ndarray,
    save: bool = True,
) -> go.Figure:
    """Bar chart of mean |SHAP value| per feature (global importance).

    Args:
        shap_vals: (n_samples, n_features) array.
        save: Whether to persist to disk.

    Returns:
        Plotly Figure.
    """
    mean_abs = np.mean(np.abs(shap_vals), axis=0)

    # Sort descending
    order = np.argsort(mean_abs)[::-1]
    features = [_FEATURE_NAMES[i] for i in order]
    importance = mean_abs[order]

    fig = go.Figure(
        go.Bar(
            y=features[::-1],
            x=importance[::-1],
            orientation="h",
            marker_color="#636EFA",
            text=[f"{v:.4f}" for v in importance[::-1]],
            textposition="outside",
        )
    )

    fig.update_layout(
        title="Global Feature Importance (Mean |SHAP Value|)",
        xaxis_title="Mean |SHAP Value|",
        yaxis_title="Feature",
        template="plotly_white",
        height=750,
        width=950,
        margin=dict(l=200),
    )

    if save:
        _save(fig, "shap_global_feature_importance")
    return fig


# ---------------------------------------------------------------------------
# Plot 2: Local Feature Importance (Single Sample)
# ---------------------------------------------------------------------------
def plot_local_feature_importance(
    shap_vals: np.ndarray,
    sample_idx: int,
    X: Optional[np.ndarray] = None,
    save: bool = True,
) -> go.Figure:
    """Horizontal bar chart of SHAP values for a single prediction.

    Args:
        shap_vals: Full (n_samples, n_features) array.
        sample_idx: Index of the sample to explain.
        X: Optional raw feature array (for display values).
        save: Whether to persist to disk.

    Returns:
        Plotly Figure.
    """
    vals = shap_vals[sample_idx]
    order = np.argsort(np.abs(vals))[::-1]

    features = [_FEATURE_NAMES[i] for i in order]
    shap_values_ordered = vals[order]
    colors = ["#EF553B" if v > 0 else "#636EFA" for v in shap_values_ordered]

    fig = go.Figure(
        go.Bar(
            y=features[::-1],
            x=shap_values_ordered[::-1],
            orientation="h",
            marker_color=colors[::-1],
            text=[f"{v:.4f}" for v in shap_values_ordered[::-1]],
            textposition="outside",
        )
    )

    fig.update_layout(
        title=f"Local Feature Importance — Sample #{sample_idx}",
        xaxis_title="SHAP Value",
        yaxis_title="Feature",
        template="plotly_white",
        height=750,
        width=950,
        margin=dict(l=200),
    )

    if save:
        _save(fig, f"shap_local_feature_importance_sample_{sample_idx}")
    return fig


# ---------------------------------------------------------------------------
# Plot 3: Waterfall Plot
# ---------------------------------------------------------------------------
def plot_waterfall(
    shap_vals: np.ndarray,
    sample_idx: int,
    save: bool = True,
) -> go.Figure:
    """Waterfall chart showing how each feature pushes the prediction
    from the base value to the output value for a single sample.

    Args:
        shap_vals: (n_samples, n_features) array.
        sample_idx: Index of the sample to explain.
        save: Whether to persist to disk.

    Returns:
        Plotly Figure.
    """
    vals = shap_vals[sample_idx]
    order = np.argsort(np.abs(vals))[::-1]

    features = [_FEATURE_NAMES[i] for i in order]
    values = vals[order]
    cumulative = np.cumsum(values)

    # Build waterfall steps
    base_value = 0.0
    y_labels = ["base"] + features + ["output"]
    x_vals = [base_value] + list(values) + [0]
    measure = ["absolute"] + ["relative"] * len(features) + ["total"]

    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=measure,
            x=y_labels,
            y=x_vals,
            textposition="outside",
            text=[f"{v:+.3f}" for v in [base_value] + list(values) + [cumulative[-1]]],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#EF553B"}},
            decreasing={"marker": {"color": "#636EFA"}},
            totals={"marker": {"color": "#00CC96"}},
        )
    )

    fig.update_layout(
        title=f"Waterfall Plot — Sample #{sample_idx}",
        yaxis_title="SHAP Value",
        xaxis_title="",
        template="plotly_white",
        height=800,
        width=1000,
        xaxis_tickangle=-45,
    )

    if save:
        _save(fig, f"shap_waterfall_sample_{sample_idx}")
    return fig


# ---------------------------------------------------------------------------
# Plot 4: Force Plot (Individual Prediction Breakdown)
# ---------------------------------------------------------------------------
def plot_force(
    shap_vals: np.ndarray,
    sample_idx: int,
    save: bool = True,
) -> go.Figure:
    """Plotly recreation of a SHAP force plot for a single prediction.

    Displays feature contributions as horizontal arrows pushing the
    prediction from the base value.

    Args:
        shap_vals: (n_samples, n_features) array.
        sample_idx: Index of the sample to explain.
        save: Whether to persist to disk.

    Returns:
        Plotly Figure.
    """
    vals = shap_vals[sample_idx]
    order = np.argsort(np.abs(vals))[::-1]

    features = [_FEATURE_NAMES[i] for i in order]
    values = vals[order]
    base = 0.0
    output = base + values.sum()

    # Build horizontal stacked bars
    positive_vals = np.where(values > 0, values, 0)
    negative_vals = np.where(values < 0, values, 0)

    fig = go.Figure()

    # Positive contributions (push higher)
    fig.add_trace(go.Bar(
        y=["SHAP Contributions"],
        x=positive_vals,
        orientation="h",
        name="Increases risk",
        marker_color="#EF553B",
        text=[f"{_FEATURE_NAMES[order[i]]}: {positive_vals[i]:+.3f}"
              for i in range(len(features)) if positive_vals[i] > 0],
        textposition="inside",
    ))

    # Negative contributions (push lower)
    fig.add_trace(go.Bar(
        y=["SHAP Contributions"],
        x=negative_vals,
        orientation="h",
        name="Decreases risk",
        marker_color="#636EFA",
        text=[f"{_FEATURE_NAMES[order[i]]}: {negative_vals[i]:+.3f}"
              for i in range(len(features)) if negative_vals[i] < 0],
        textposition="inside",
    ))

    fig.update_layout(
        barmode="relative",
        title=(
            f"Force Plot — Sample #{sample_idx} "
            f"(base={base:.3f} → output={output:.3f})"
        ),
        template="plotly_white",
        height=300,
        width=1000,
        showlegend=True,
        legend=dict(orientation="h", y=1.15),
    )

    # Add annotation listing top features
    top_n = 8
    top_features = [
        f"{_FEATURE_NAMES[order[i]]}: {values[i]:+.3f}"
        for i in range(min(top_n, len(features)))
    ]
    fig.add_annotation(
        text="<b>Top features:</b> " + " | ".join(top_features),
        xref="paper", yref="paper",
        x=0.5, y=-0.15,
        showarrow=False,
        font=dict(size=10),
    )

    if save:
        _save(fig, f"shap_force_plot_sample_{sample_idx}")
    return fig


# ---------------------------------------------------------------------------
# Plot 5: Summary Plot (Beeswarm-style)
# ---------------------------------------------------------------------------
def plot_summary(
    shap_vals: np.ndarray,
    save: bool = True,
) -> go.Figure:
    """Scatter-based beeswarm summary of all SHAP values across features.

    Each dot represents one sample's SHAP value for one feature.
    Dots are coloured by the feature value (high vs low).

    Args:
        shap_vals: (n_samples, n_features) array.
        save: Whether to persist to disk.

    Returns:
        Plotly Figure.
    """
    n_samples, n_features = shap_vals.shape
    order = np.mean(np.abs(shap_vals), axis=0).argsort()[::-1]

    fig = go.Figure()

    for rank, feat_idx in enumerate(order):
        feat_name = _FEATURE_NAMES[feat_idx]
        vals = shap_vals[:, feat_idx]

        # Jitter y-position for visibility
        jitter = np.random.RandomState(42).uniform(-0.15, 0.15, size=n_samples)

        fig.add_trace(go.Scatter(
            x=vals,
            y=[rank + j for j in jitter],
            mode="markers",
            marker=dict(
                size=4,
                color=vals,
                colorscale="RdBu_r",
                showscale=(rank == 0),
                colorbar=dict(title="SHAP Value") if rank == 0 else None,
                opacity=0.6,
            ),
            name=feat_name,
            hovertemplate=(
                f"<b>{feat_name}</b><br>"
                "SHAP Value: %{x:.4f}<br>"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="SHAP Summary Plot (Beeswarm)",
        xaxis_title="SHAP Value (impact on model output)",
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(n_features)),
            ticktext=[_FEATURE_NAMES[i] for i in order],
        ),
        template="plotly_white",
        height=850,
        width=1000,
        showlegend=False,
    )

    if save:
        _save(fig, "shap_summary_plot")
    return fig


# ---------------------------------------------------------------------------
# Utility: Save Plotly figure
# ---------------------------------------------------------------------------
def _save(fig: go.Figure, name: str) -> None:
    """Write a Plotly figure to HTML."""
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = _FIGURES_DIR / f"{name}.html"
    fig.write_html(str(path), include_plotlyjs="cdn")
    logger.info("Saved: %s", path.name)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_explainability(
    background_samples: int = 100,
    n_test: int = 50,
    sample_idx: int = 0,
) -> None:
    """Execute the full explainability pipeline.

    Steps:
        1. Load model, background data, and test data.
        2. Build SHAP DeepExplainer.
        3. Compute SHAP values.
        4. Generate all five plots.
        5. Print global importance rankings.

    Args:
        background_samples: Number of background samples for SHAP.
        n_test: Number of test samples to explain.
        sample_idx: Index into test set for local explanations.
    """
    logger.info("=" * 60)
    logger.info("STARTING SHAP EXPLAINABILITY PIPELINE")
    logger.info("=" * 60)

    # 1. Load
    model = load_model()
    background = load_background_data(n_samples=background_samples)
    X_test, y_test = load_test_data()

    # Limit test set for speed
    if n_test and n_test < X_test.shape[0]:
        X_test = X_test[:n_test]
        y_test = y_test[:n_test]

    logger.info("Test subset for explanation: %s", X_test.shape)

    # 2. Build explainer
    explainer = build_shap_explainer(model, background)

    # 3. Compute SHAP values
    shap_vals = compute_shap_values(explainer, X_test)

    # 4. Generate plots
    logger.info("Generating plots …")

    # 4a. Global feature importance
    plot_global_feature_importance(shap_vals)

    # 4b. Local feature importance
    plot_local_feature_importance(shap_vals, sample_idx)

    # 4c. Waterfall plot
    plot_waterfall(shap_vals, sample_idx)

    # 4d. Force plot
    plot_force(shap_vals, sample_idx)

    # 4e. Summary plot
    plot_summary(shap_vals)

    # 5. Print global rankings
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    ranked = sorted(
        zip(_FEATURE_NAMES, mean_abs), key=lambda x: x[1], reverse=True
    )

    logger.info("\n" + "=" * 50)
    logger.info("GLOBAL FEATURE IMPORTANCE RANKING")
    logger.info("=" * 50)
    for rank, (feat, imp) in enumerate(ranked, 1):
        bar = "#" * int(imp / ranked[0][1] * 30)
        logger.info("  %2d. %-30s %.4f  %s", rank, feat, imp, bar)

    logger.info("=" * 60)
    logger.info("SHAP EXPLAINABILITY PIPELINE COMPLETE")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_explainability()
