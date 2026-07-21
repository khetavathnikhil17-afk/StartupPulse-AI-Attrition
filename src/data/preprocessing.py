"""
Preprocessing Pipeline for StartupPulse AI – Employee Attrition Prediction.

This module implements a production-ready, end-to-end preprocessing pipeline:
    1. Load raw dataset
    2. Remove duplicate records
    3. Handle missing values
    4. Encode categorical columns via LabelEncoder
    5. Map Attrition target (Yes→1, No→0)
    6. Scale numerical features via StandardScaler
    7. Split into train / validation / test (70 / 15 / 15)
    8. Persist processed splits and fitted artefacts
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
_RAW_DATA_PATH = _ROOT / "data" / "raw" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
_PROCESSED_DIR = _ROOT / "data" / "processed"
_ARTEFACTS_DIR = _ROOT / "models" / "startuppulse_v1"

# Columns that are identifiers or constants (carry no predictive signal)
_COLS_TO_DROP: List[str] = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
def load_raw_data(path: Path = _RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV into a DataFrame.

    Args:
        path: Filesystem path to the raw CSV.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    logger.info("Loading raw data from %s", path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")
    df = pd.read_csv(path)
    logger.info("Raw data loaded – shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# 2. Remove duplicates
# ---------------------------------------------------------------------------
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows (first occurrence kept).

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with duplicates removed.
    """
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_removed = n_before - len(df)
    logger.info("Duplicates removed: %d (rows %d → %d)", n_removed, n_before, len(df))
    return df


# ---------------------------------------------------------------------------
# 3. Handle missing values
# ---------------------------------------------------------------------------
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values.

    Strategy:
        - Numeric columns → median
        - Categorical columns → mode (most frequent value)

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with no missing values.
    """
    n_missing_total = df.isnull().sum().sum()
    if n_missing_total == 0:
        logger.info("No missing values detected.")
        return df

    logger.info("Handling %d missing values …", n_missing_total)

    for col in df.columns:
        n_miss = df[col].isnull().sum()
        if n_miss == 0:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info("  %-35s (numeric)  → median = %s  (%d filled)", col, median_val, n_miss)
        else:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            logger.info("  %-35s (categorical) → mode  = %s  (%d filled)", col, mode_val, n_miss)

    logger.info("All missing values handled.")
    return df


# ---------------------------------------------------------------------------
# 4. Drop non-informative columns
# ---------------------------------------------------------------------------
def drop_columns(df: pd.DataFrame, columns: List[str] = _COLS_TO_DROP) -> pd.DataFrame:
    """Remove columns that carry no predictive signal.

    Args:
        df: Input DataFrame.
        columns: List of column names to drop.

    Returns:
        DataFrame with specified columns removed.
    """
    existing = [c for c in columns if c in df.columns]
    df = df.drop(columns=existing, axis=1)
    logger.info("Dropped columns: %s", existing)
    return df


# ---------------------------------------------------------------------------
# 5. Encode categorical features
# ---------------------------------------------------------------------------
def encode_categoricals(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Label-encode every categorical column.

    Args:
        df: Input DataFrame.

    Returns:
        A tuple of (encoded DataFrame, {column_name: fitted LabelEncoder}).
    """
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    logger.info("Categorical columns to encode (%d): %s", len(cat_cols), cat_cols)

    encoders: Dict[str, LabelEncoder] = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        logger.info("  Encoded %-35s → %d classes", col, len(le.classes_))

    return df, encoders


# ---------------------------------------------------------------------------
# 6. Map Attrition target
# ---------------------------------------------------------------------------
def map_attrition(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure Attrition is numeric 1 (Yes) / 0 (No).

    If the column has already been label-encoded, verify the mapping.
    Otherwise, apply an explicit Yes/No → 1/0 mapping.

    Args:
        df: Input DataFrame with an 'Attrition' column.

    Returns:
        DataFrame with Attrition values guaranteed to be 0 or 1.
    """
    if df["Attrition"].dtype == object:
        mapping = {"Yes": 1, "No": 0}
        df["Attrition"] = df["Attrition"].map(mapping).astype(int)
        logger.info("Mapped Attrition: Yes→1, No→0")
    else:
        # Already numeric – just ensure int
        df["Attrition"] = df["Attrition"].astype(int)
        logger.info("Attrition column already numeric – cast to int.")

    dist = df["Attrition"].value_counts().to_dict()
    logger.info("Attrition distribution: %s", dist)
    return df


# ---------------------------------------------------------------------------
# 7. Scale numerical features
# ---------------------------------------------------------------------------
def scale_numericals(
    df: pd.DataFrame,
    exclude_cols: List[str] = None,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """Apply StandardScaler to all numeric columns.

    Args:
        df: Input DataFrame.
        exclude_cols: Column names to skip (e.g., the target).

    Returns:
        A tuple of (scaled DataFrame, fitted StandardScaler).
    """
    exclude = set(exclude_cols or ["Attrition"])
    num_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude
    ]
    logger.info("Numerical columns to scale (%d): %s", len(num_cols), num_cols)

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    logger.info("StandardScaler applied to %d columns.", len(num_cols))
    return df, scaler


# ---------------------------------------------------------------------------
# 8. Train / Validation / Test split
# ---------------------------------------------------------------------------
def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train, validation, and test sets.

    Uses a two-stage stratified split so class proportions are preserved.

    Args:
        df: Full preprocessed DataFrame.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        random_state: Reproducibility seed.

    Returns:
        (train_df, val_df, test_df)

    Raises:
        ValueError: If ratios do not sum to 1.0.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        )

    X = df.drop(columns=["Attrition"])
    y = df["Attrition"]

    # Stage 1: hold out test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=test_ratio,
        random_state=random_state,
        stratify=y,
    )

    # Stage 2: split remaining into train and validation
    relative_val = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=relative_val,
        random_state=random_state,
        stratify=y_trainval,
    )

    train_df = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    val_df = pd.concat([X_val, y_val], axis=1).reset_index(drop=True)
    test_df = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)

    logger.info(
        "Split complete – train: %d | val: %d | test: %d",
        len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# 9. Persist artefacts
# ---------------------------------------------------------------------------
def save_processed_data(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Path = _PROCESSED_DIR,
) -> None:
    """Save train / val / test splits to CSV.

    Args:
        train: Training DataFrame.
        val: Validation DataFrame.
        test: Test DataFrame.
        output_dir: Target directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "train.csv", index=False)
    val.to_csv(output_dir / "validation.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)
    logger.info("Processed splits saved to %s", output_dir)


def save_scaler(scaler: StandardScaler, output_dir: Path = _ARTEFACTS_DIR) -> None:
    """Persist the fitted StandardScaler to disk.

    Args:
        scaler: Fitted StandardScaler instance.
        output_dir: Target directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "scaler.pkl"
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Scaler saved to %s", path)


def save_label_encoders(
    encoders: Dict[str, LabelEncoder],
    output_dir: Path = _ARTEFACTS_DIR,
) -> None:
    """Persist the fitted LabelEncoders to disk.

    Args:
        encoders: Dict mapping column names to fitted LabelEncoder instances.
        output_dir: Target directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "label_encoders.pkl"
    with open(path, "wb") as f:
        pickle.dump(encoders, f)
    logger.info("Label encoders saved to %s", path)


# ---------------------------------------------------------------------------
# 10. Full pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    raw_path: Path = _RAW_DATA_PATH,
    processed_dir: Path = _PROCESSED_DIR,
    artefacts_dir: Path = _ARTEFACTS_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute the full preprocessing pipeline.

    Steps:
        1. Load raw data
        2. Remove duplicates
        3. Handle missing values
        4. Drop non-informative columns
        5. Encode categorical features
        6. Map Attrition to 0/1
        7. Scale numerical features
        8. Split into train / validation / test
        9. Save processed splits and artefacts

    Args:
        raw_path: Path to the raw CSV file.
        processed_dir: Directory for processed splits.
        artefacts_dir: Directory for scaler and encoders.

    Returns:
        (train_df, val_df, test_df)
    """
    logger.info("=" * 60)
    logger.info("STARTING PREPROCESSING PIPELINE")
    logger.info("=" * 60)

    # 1. Load
    df = load_raw_data(raw_path)

    # 2. Duplicates
    df = remove_duplicates(df)

    # 3. Missing values
    df = handle_missing_values(df)

    # 4. Drop useless columns
    df = drop_columns(df)

    # 5. Encode categoricals
    df, encoders = encode_categoricals(df)

    # 6. Map target
    df = map_attrition(df)

    # 7. Scale numerics
    df, scaler = scale_numericals(df, exclude_cols=["Attrition"])

    # 8. Split
    train, val, test = split_data(df)

    # 9. Save
    save_processed_data(train, val, test, processed_dir)
    save_scaler(scaler, artefacts_dir)
    save_label_encoders(encoders, artefacts_dir)

    logger.info("=" * 60)
    logger.info("PREPROCESSING PIPELINE COMPLETE")
    logger.info("=" * 60)

    return train, val, test


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    train_df, val_df, test_df = run_pipeline()

    logger.info("\n--- Summary ---")
    logger.info("Train shape      : %s", train_df.shape)
    logger.info("Validation shape : %s", val_df.shape)
    logger.info("Test shape       : %s", test_df.shape)
