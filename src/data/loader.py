"""Data loading and schema validation module for FraudShield."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from src.utils.config import get_project_root, load_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

EXPECTED_FEATURES: List[str] = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
    "Class",
]


def get_default_data_path() -> Path:
    """Returns the configured path to the raw credit card dataset."""
    config = load_config()
    raw_rel_path = config.get("data", {}).get("raw_path", "data/raw/creditcard.csv")
    return get_project_root() / raw_rel_path


def load_raw_data(filepath: Optional[Path | str] = None) -> pd.DataFrame:
    """Loads raw credit card transaction data from a CSV file.

    Args:
        filepath: Path to the CSV file. If None, uses the path from config.yaml.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the raw data file does not exist.
        ValueError: If file is empty or fails schema validation.
    """
    path = Path(filepath) if filepath is not None else get_default_data_path()

    if not path.exists():
        err_msg = (
            f"Dataset not found at: {path}\n"
            "Please ensure 'creditcard.csv' is placed inside 'data/raw/'.\n"
            "Dataset source: Kaggle Credit Card Fraud Detection benchmark dataset."
        )
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    logger.info(f"Loading raw dataset from: {path}")
    df = pd.read_csv(path)
    logger.info(f"Successfully loaded dataset with shape: {df.shape}")

    validate_schema(df)
    return df


def validate_schema(df: pd.DataFrame) -> bool:
    """Validates that the DataFrame contains expected columns and non-empty rows.

    Args:
        df: Input DataFrame.

    Returns:
        bool: True if schema is valid.

    Raises:
        ValueError: If dataframe is empty or missing expected columns.
    """
    if df.empty:
        raise ValueError("DataFrame is empty.")

    missing_cols = [col for col in EXPECTED_FEATURES if col not in df.columns]
    if missing_cols:
        err_msg = f"Dataset is missing required columns: {missing_cols}"
        logger.error(err_msg)
        raise ValueError(err_msg)

    logger.debug("Schema validation passed successfully.")
    return True


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Computes key exploratory statistics on the dataset.

    Args:
        df: Input DataFrame conforming to the credit card schema.

    Returns:
        dict: Summary statistics including class breakdown and missing values.
    """
    total_transactions = len(df)
    class_counts = df["Class"].value_counts().to_dict()
    genuine_count = int(class_counts.get(0, 0))
    fraud_count = int(class_counts.get(1, 0))

    fraud_percentage = (fraud_count / total_transactions * 100) if total_transactions > 0 else 0.0
    imbalance_ratio = (genuine_count / fraud_count) if fraud_count > 0 else float("inf")
    missing_values = int(df.isnull().sum().sum())

    amount_genuine = df[df["Class"] == 0]["Amount"]
    amount_fraud = df[df["Class"] == 1]["Amount"]

    summary = {
        "total_transactions": total_transactions,
        "genuine_count": genuine_count,
        "fraud_count": fraud_count,
        "fraud_percentage": round(fraud_percentage, 4),
        "imbalance_ratio": round(imbalance_ratio, 2),
        "missing_values": missing_values,
        "amount_summary": {
            "genuine_mean": float(amount_genuine.mean()) if not amount_genuine.empty else 0.0,
            "genuine_median": float(amount_genuine.median()) if not amount_genuine.empty else 0.0,
            "genuine_max": float(amount_genuine.max()) if not amount_genuine.empty else 0.0,
            "fraud_mean": float(amount_fraud.mean()) if not amount_fraud.empty else 0.0,
            "fraud_median": float(amount_fraud.median()) if not amount_fraud.empty else 0.0,
            "fraud_max": float(amount_fraud.max()) if not amount_fraud.empty else 0.0,
        },
    }

    return summary

