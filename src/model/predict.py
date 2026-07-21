"""
Prediction Pipeline – StartupPulse AI: Employee Attrition Prediction.

Provides a production-ready interface to load the trained model,
preprocess raw employee records, and return attrition predictions
with probability scores, risk levels, and recommended HR actions.
"""

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tensorflow import keras

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = _ROOT / "models" / "startuppulse_v1"

# Canonical feature order (must match training data exactly)
_FEATURE_ORDER: List[str] = [
    "Age", "BusinessTravel", "DailyRate", "Department", "DistanceFromHome",
    "Education", "EducationField", "EnvironmentSatisfaction", "Gender",
    "HourlyRate", "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction",
    "MaritalStatus", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked",
    "OverTime", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
]

# Categorical columns that require label encoding
_CATEGORICAL_COLS: List[str] = [
    "BusinessTravel", "Department", "EducationField", "Gender",
    "JobRole", "MaritalStatus", "OverTime",
]

# Numeric columns (all remaining after categoricals)
_NUMERIC_COLS: List[str] = [c for c in _FEATURE_ORDER if c not in _CATEGORICAL_COLS]


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class PredictionResult:
    """Structured result returned by the prediction pipeline."""

    prediction: str
    probability: float
    risk_level: str
    recommended_action: str
    confidence: float
    raw_probability: float = field(repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary."""
        return {
            "prediction": self.prediction,
            "probability": round(self.probability, 4),
            "risk_level": self.risk_level,
            "recommended_action": self.recommended_action,
            "confidence": round(self.confidence, 4),
        }

    def summary(self) -> str:
        """Human-readable one-line summary."""
        return (
            f"Prediction: {self.prediction} "
            f"(prob={self.probability:.2%}, risk={self.risk_level})"
        )


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------
def load_model(
    model_path: Path = _MODEL_DIR / "attrition_model.keras",
) -> keras.Model:
    """Load the trained Keras model from disk.

    Args:
        model_path: Path to the saved .keras model file.

    Returns:
        A compiled Keras Model.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = keras.models.load_model(model_path)
    logger.info("Model loaded from %s", model_path)
    return model


def load_scaler(
    scaler_path: Path = _MODEL_DIR / "scaler.pkl",
) -> Any:
    """Load the fitted StandardScaler from disk.

    Args:
        scaler_path: Path to the saved scaler pickle.

    Returns:
        A fitted sklearn StandardScaler instance.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    logger.info("Scaler loaded from %s", scaler_path)
    return scaler


def load_label_encoders(
    encoders_path: Path = _MODEL_DIR / "label_encoders.pkl",
) -> Dict[str, Any]:
    """Load the fitted LabelEncoders from disk.

    Args:
        encoders_path: Path to the saved encoders pickle.

    Returns:
        Dict mapping column names to fitted LabelEncoder instances.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not encoders_path.exists():
        raise FileNotFoundError(f"Label encoders not found: {encoders_path}")
    with open(encoders_path, "rb") as f:
        encoders = pickle.load(f)
    logger.info("Label encoders loaded (%d cols): %s", len(encoders), list(encoders.keys()))
    return encoders


# ---------------------------------------------------------------------------
# Input Preprocessing
# ---------------------------------------------------------------------------
def preprocess_input(
    raw_input: Dict[str, Any],
    scaler: Any,
    label_encoders: Dict[str, Any],
) -> np.ndarray:
    """Transform a raw employee record into a scaled feature vector.

    Steps:
        1. Validate that all required keys are present.
        2. Encode categorical columns using the fitted LabelEncoders.
        3. Arrange features in canonical order.
        4. Scale all numeric values using the fitted StandardScaler.

    Args:
        raw_input: Dictionary with human-readable feature values.
            Categorical columns should use the original string labels
            (e.g., BusinessTravel='Travel_Rarely', OverTime='Yes').
            Numeric columns should be raw integers or floats.
        scaler: Fitted StandardScaler.
        label_encoders: Dict of fitted LabelEncoders.

    Returns:
        A (1, 30) NumPy float32 array ready for model.predict().

    Raises:
        ValueError: If required keys are missing or an unknown
            categorical label is encountered.
    """
    # --- 1. Validate keys ---
    missing = [col for col in _FEATURE_ORDER if col not in raw_input]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    # --- 2. Encode categoricals ---
    encoded_row: Dict[str, Any] = {}
    for col in _FEATURE_ORDER:
        value = raw_input[col]

        if col in _CATEGORICAL_COLS:
            le = label_encoders[col]
            value_str = str(value)

            if value_str not in le.classes_:
                raise ValueError(
                    f"Unknown label for '{col}': '{value_str}'. "
                    f"Valid labels: {list(le.classes_)}"
                )
            encoded_row[col] = le.transform([value_str])[0]
        else:
            encoded_row[col] = value

    # --- 3. Arrange in canonical order ---
    feature_vector = np.array(
        [[encoded_row[col] for col in _FEATURE_ORDER]],
        dtype=np.float32,
    )

    # --- 4. Scale ---
    feature_vector = scaler.transform(feature_vector)

    logger.debug("Preprocessed input → shape %s", feature_vector.shape)
    return feature_vector


# ---------------------------------------------------------------------------
# Risk Assessment
# ---------------------------------------------------------------------------
def _compute_risk_level(probability: float) -> str:
    """Map the attrition probability to a risk category.

    Thresholds:
        - probability < 0.30 → Low Risk
        - 0.30 <= probability < 0.60 → Medium Risk
        - probability >= 0.60 → High Risk

    Args:
        probability: Model-predicted probability of attrition.

    Returns:
        One of 'Low Risk', 'Medium Risk', 'High Risk'.
    """
    if probability < 0.30:
        return "Low Risk"
    elif probability < 0.60:
        return "Medium Risk"
    return "High Risk"


def _recommend_action(risk_level: str, probability: float) -> str:
    """Generate a recommended HR action based on the risk level.

    Args:
        risk_level: Risk category string.
        probability: Raw attrition probability.

    Returns:
        A specific, actionable HR recommendation.
    """
    if risk_level == "Low Risk":
        return (
            "No immediate action required. Continue standard engagement "
            "activities and periodic check-ins."
        )
    elif risk_level == "Medium Risk":
        return (
            "Schedule a one-on-one meeting to discuss career growth, "
            "compensation review, and job satisfaction. Monitor closely "
            "over the next 30 days."
        )
    return (
        "Urgent intervention required. Initiate a retention conversation, "
        "consider a compensation adjustment or role realignment, and "
        "escalate to the department head if risk persists above 80%."
    )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_attrition(
    raw_input: Dict[str, Any],
    model: Optional[keras.Model] = None,
    scaler: Optional[Any] = None,
    label_encoders: Optional[Dict[str, Any]] = None,
    threshold: float = 0.5,
) -> PredictionResult:
    """Run the full prediction pipeline for a single employee record.

    This is the primary public interface. It orchestrates:
        1. Lazy-loading of model / scaler / encoders (if not provided).
        2. Input preprocessing.
        3. Model inference.
        4. Risk assessment and action recommendation.

    Args:
        raw_input: Dictionary of raw feature values.
            Example:
                {
                    "Age": 41,
                    "BusinessTravel": "Travel_Rarely",
                    "DailyRate": 1102,
                    "Department": "Sales",
                    "DistanceFromHome": 1,
                    ...
                }
        model: Pre-loaded Keras model (optional; loaded on demand).
        scaler: Pre-loaded StandardScaler (optional; loaded on demand).
        label_encoders: Pre-loaded LabelEncoders dict (optional).
        threshold: Classification threshold (default 0.5).

    Returns:
        A PredictionResult with prediction, probability, risk level,
        and recommended action.

    Raises:
        FileNotFoundError: If artefact files are missing.
        ValueError: If input validation fails.
    """
    logger.info("Running attrition prediction …")

    # --- Lazy-load artefacts ---
    if model is None:
        model = load_model()
    if scaler is None:
        scaler = load_scaler()
    if label_encoders is None:
        label_encoders = load_label_encoders()

    # --- Preprocess ---
    X = preprocess_input(raw_input, scaler, label_encoders)

    # --- Predict ---
    prob = float(model.predict(X, verbose=0)[0, 0])
    pred_label = 1 if prob >= threshold else 0

    # --- Map output ---
    prediction = "Likely to Leave" if pred_label == 1 else "Likely to Stay"
    risk_level = _compute_risk_level(prob)
    action = _recommend_action(risk_level, prob)
    confidence = prob if pred_label == 1 else 1.0 - prob

    result = PredictionResult(
        prediction=prediction,
        probability=prob,
        risk_level=risk_level,
        recommended_action=action,
        confidence=confidence,
        raw_probability=prob,
    )

    logger.info(result.summary())
    return result


def predict_batch(
    records: List[Dict[str, Any]],
    model: Optional[keras.Model] = None,
    scaler: Optional[Any] = None,
    label_encoders: Optional[Dict[str, Any]] = None,
    threshold: float = 0.5,
) -> List[PredictionResult]:
    """Run predictions on multiple employee records at once.

    Args:
        records: List of raw feature dictionaries.
        model: Pre-loaded Keras model (optional).
        scaler: Pre-loaded StandardScaler (optional).
        label_encoders: Pre-loaded LabelEncoders dict (optional).
        threshold: Classification threshold (default 0.5).

    Returns:
        List of PredictionResult objects, one per record.
    """
    # Lazy-load once
    if model is None:
        model = load_model()
    if scaler is None:
        scaler = load_scaler()
    if label_encoders is None:
        label_encoders = load_label_encoders()

    results: List[PredictionResult] = []
    for idx, record in enumerate(records):
        logger.info("Predicting record %d / %d", idx + 1, len(records))
        result = predict_attrition(
            raw_input=record,
            model=model,
            scaler=scaler,
            label_encoders=label_encoders,
            threshold=threshold,
        )
        results.append(result)

    # Summary stats
    n_leave = sum(1 for r in results if r.prediction == "Likely to Leave")
    logger.info(
        "Batch prediction complete — %d records, %d predicted to leave",
        len(results), n_leave,
    )
    return results


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sample_employee = {
        "Age": 41,
        "BusinessTravel": "Travel_Rarely",
        "DailyRate": 1102,
        "Department": "Sales",
        "DistanceFromHome": 1,
        "Education": 2,
        "EducationField": "Life Sciences",
        "EnvironmentSatisfaction": 2,
        "Gender": "Female",
        "HourlyRate": 94,
        "JobInvolvement": 3,
        "JobLevel": 2,
        "JobRole": "Sales Executive",
        "JobSatisfaction": 4,
        "MaritalStatus": "Single",
        "MonthlyIncome": 5993,
        "MonthlyRate": 19479,
        "NumCompaniesWorked": 8,
        "OverTime": "Yes",
        "PercentSalaryHike": 11,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 1,
        "StockOptionLevel": 0,
        "TotalWorkingYears": 8,
        "TrainingTimesLastYear": 0,
        "WorkLifeBalance": 1,
        "YearsAtCompany": 6,
        "YearsInCurrentRole": 4,
        "YearsSinceLastPromotion": 0,
        "YearsWithCurrManager": 5,
    }

    result = predict_attrition(sample_employee)

    logger.info("=" * 60)
    logger.info("PREDICTION RESULT")
    logger.info("=" * 60)
    for key, value in result.to_dict().items():
        logger.info("  %-25s: %s", key, value)
    logger.info("=" * 60)
