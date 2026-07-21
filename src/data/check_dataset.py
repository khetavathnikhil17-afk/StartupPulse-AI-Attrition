"""
Dataset Exploration & Validation Module.

Provides functions to load the raw HR Employee Attrition dataset and
print a comprehensive summary including shape, column names, data types,
missing values, and duplicate rows.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_RAW_DATA_PATH = _ROOT / "data" / "raw" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"


def load_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the raw CSV dataset into a pandas DataFrame.

    Args:
        path: Path to the CSV file. Defaults to the project's raw data path.

    Returns:
        A pandas DataFrame containing the raw dataset.

    Raises:
        FileNotFoundError: If the CSV file does not exist at *path*.
        pd.errors.EmptyDataError: If the file is empty.
    """
    data_path = path or _RAW_DATA_PATH
    logger.info("Loading dataset from %s", data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    logger.info("Dataset loaded successfully – shape: %s", df.shape)
    return df


def check_dataset(df: pd.DataFrame) -> None:
    """Print a full exploratory summary of the dataset.

    Reports:
        - Dataset shape (rows x columns)
        - Column names
        - Data types
        - Missing values per column
        - Total duplicate rows
    """
    logger.info("=" * 60)
    logger.info("DATASET EXPLORATION REPORT")
    logger.info("=" * 60)

    # --- Shape ---
    logger.info("\n[1] Dataset Shape: %s rows x %s columns", *df.shape)

    # --- Column Names ---
    logger.info("\n[2] Column Names:")
    for idx, col in enumerate(df.columns, start=1):
        logger.info("    %2d. %s", idx, col)

    # --- Data Types ---
    logger.info("\n[3] Data Types:")
    for col, dtype in df.dtypes.items():
        logger.info("    %-35s -> %s", col, dtype)

    # --- Missing Values ---
    missing = df.isnull().sum()
    total_missing = missing.sum()
    logger.info("\n[4] Missing Values (total: %d):", total_missing)
    if total_missing == 0:
        logger.info("    No missing values found.")
    else:
        for col, count in missing[missing > 0].items():
            pct = (count / len(df)) * 100
            logger.info("    %-35s -> %d (%.2f%%)", col, count, pct)

    # --- Duplicate Rows ---
    n_duplicates = df.duplicated().sum()
    logger.info("\n[5] Duplicate Rows: %d", n_duplicates)
    if n_duplicates == 0:
        logger.info("    No duplicate records found.")
    else:
        logger.info("    %.2f%% of the dataset consists of duplicates.", (n_duplicates / len(df)) * 100)

    logger.info("\n" + "=" * 60)
    logger.info("END OF REPORT")
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    df = load_dataset()
    check_dataset(df)
