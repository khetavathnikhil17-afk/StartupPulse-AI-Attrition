"""
Configuration Module – StartupPulse AI.

Centralizes all path constants, feature definitions, and project-wide
settings so that every module references a single source of truth.
"""

from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------
ROOT: Path = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Data Paths
# ---------------------------------------------------------------------------
DATA_RAW_DIR: Path = ROOT / "data" / "raw"
DATA_PROCESSED_DIR: Path = ROOT / "data" / "processed"
RAW_DATASET_PATH: Path = DATA_RAW_DIR / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
TRAIN_PATH: Path = DATA_PROCESSED_DIR / "train.csv"
VALIDATION_PATH: Path = DATA_PROCESSED_DIR / "validation.csv"
TEST_PATH: Path = DATA_PROCESSED_DIR / "test.csv"

# ---------------------------------------------------------------------------
# Model Paths
# ---------------------------------------------------------------------------
MODEL_DIR: Path = ROOT / "models" / "startuppulse_v1"
MODEL_PATH: Path = MODEL_DIR / "attrition_model.keras"
SCALER_PATH: Path = MODEL_DIR / "scaler.pkl"
ENCODERS_PATH: Path = MODEL_DIR / "label_encoders.pkl"

# ---------------------------------------------------------------------------
# Report Paths
# ---------------------------------------------------------------------------
FIGURES_DIR: Path = ROOT / "reports" / "figures"
RESULTS_DIR: Path = ROOT / "reports" / "results"
METRICS_PATH: Path = RESULTS_DIR / "metrics.json"
EVAL_REPORT_PATH: Path = RESULTS_DIR / "evaluation_report.md"
EDA_REPORT_PATH: Path = ROOT / "reports" / "eda_summary_report.md"

# ---------------------------------------------------------------------------
# Feature Definitions
# ---------------------------------------------------------------------------
FEATURE_ORDER: List[str] = [
    "Age", "BusinessTravel", "DailyRate", "Department", "DistanceFromHome",
    "Education", "EducationField", "EnvironmentSatisfaction", "Gender",
    "HourlyRate", "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction",
    "MaritalStatus", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked",
    "OverTime", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
]

CATEGORICAL_COLS: List[str] = [
    "BusinessTravel", "Department", "EducationField", "Gender",
    "JobRole", "MaritalStatus", "OverTime",
]

NUMERIC_COLS: List[str] = [c for c in FEATURE_ORDER if c not in CATEGORICAL_COLS]

DROPPED_COLS: List[str] = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]

# ---------------------------------------------------------------------------
# Training Hyper-parameters
# ---------------------------------------------------------------------------
EPOCHS: int = 100
BATCH_SIZE: int = 32
LEARNING_RATE: float = 0.001
PATIENCE_EARLY_STOPPING: int = 10
PATIENCE_LR_PLATEAU: int = 5
MIN_LR: float = 1e-6
VALIDATION_SPLIT: float = 0.0
TEST_SIZE: float = 0.15
VAL_SIZE: float = 0.15

# ---------------------------------------------------------------------------
# Prediction Thresholds
# ---------------------------------------------------------------------------
RISK_THRESHOLDS: Dict[str, float] = {
    "low": 0.30,
    "medium": 0.60,
}

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
DASHBOARD_TITLE: str = "StartupPulse AI"
DASHBOARD_ICON: str = "\U0001f9e0"
DASHBOARD_VERSION: str = "1.0.0"
