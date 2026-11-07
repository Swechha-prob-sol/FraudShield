"""Data loading, memory optimization, and schema validation module for FraudShield."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.utils.config import get_project_root, load_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Legacy Kaggle European dataset columns
EXPECTED_FEATURES_LEGACY: List[str] = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
    "Class",
]

# IEEE-CIS Fraud Detection core columns
IEEE_CORE_COLUMNS: List[str] = [
    "TransactionID",
    "isFraud",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
]


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Downcasts numeric columns in a pandas DataFrame to reduce RAM consumption.

    Args:
        df: Input DataFrame.
        verbose: Whether to log memory savings.

    Returns:
        pd.DataFrame: Memory-optimized DataFrame.
    """
    start_mem = df.memory_usage().sum() / (1024**2)

    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        elif pd.api.types.is_float_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage().sum() / (1024**2)
    if verbose:
        reduction = (start_mem - end_mem) / start_mem * 100 if start_mem > 0 else 0
        logger.info(
            f"Memory reduced from {start_mem:.2f} MB to {end_mem:.2f} MB "
            f"({reduction:.1f}% reduction)"
        )

    return df


def get_default_paths() -> Dict[str, Path]:
    """Returns resolved paths to raw data files configured in config.yaml."""
    config = load_config()
    root = get_project_root()
    data_cfg = config.get("data", {})

    return {
        "transaction": root / data_cfg.get("raw_transaction_path", "data/raw/train_transaction.csv"),
        "identity": root / data_cfg.get("raw_identity_path", "data/raw/train_identity.csv"),
        "legacy": root / data_cfg.get("raw_path", "data/raw/creditcard.csv"),
    }


def load_ieee_data(
    transaction_path: Optional[Path | str] = None,
    identity_path: Optional[Path | str] = None,
    nrows: Optional[int] = None,
    optimize_mem: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads the IEEE-CIS transaction and identity datasets.

    Args:
        transaction_path: Path to train_transaction.csv.
        identity_path: Path to train_identity.csv.
        nrows: Number of rows to read (useful for rapid exploration).
        optimize_mem: Whether to downcast dtypes to conserve RAM.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (df_transaction, df_identity)

    Raises:
        FileNotFoundError: If the transaction or identity file does not exist.
    """
    paths = get_default_paths()
    tr_path = Path(transaction_path) if transaction_path else paths["transaction"]
    id_path = Path(identity_path) if identity_path else paths["identity"]

    if not tr_path.exists():
        raise FileNotFoundError(f"Transaction dataset not found at: {tr_path}")
    if not id_path.exists():
        raise FileNotFoundError(f"Identity dataset not found at: {id_path}")

    logger.info(f"Loading IEEE transaction data from: {tr_path} (nrows={nrows})")
    df_transaction = pd.read_csv(tr_path, nrows=nrows)

    logger.info(f"Loading IEEE identity data from: {id_path} (nrows={nrows})")
    df_identity = pd.read_csv(id_path, nrows=nrows)

    if optimize_mem:
        df_transaction = reduce_mem_usage(df_transaction, verbose=True)
        df_identity = reduce_mem_usage(df_identity, verbose=True)

    return df_transaction, df_identity


def get_ieee_summary(
    df_transaction: pd.DataFrame,
    df_identity: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Computes key summary statistics on the IEEE-CIS transaction and identity datasets."""
    total_transactions = len(df_transaction)
    target_col = "isFraud" if "isFraud" in df_transaction.columns else "Class"

    class_counts = df_transaction[target_col].value_counts().to_dict()
    genuine_count = int(class_counts.get(0, 0))
    fraud_count = int(class_counts.get(1, 0))

    fraud_pct = (fraud_count / total_transactions * 100) if total_transactions > 0 else 0.0
    imbalance_ratio = (genuine_count / fraud_count) if fraud_count > 0 else float("inf")

    num_cols = df_transaction.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df_transaction.select_dtypes(exclude=["number"]).columns.tolist()

    amt_col = "TransactionAmt" if "TransactionAmt" in df_transaction.columns else "Amount"
    amt_genuine = df_transaction[df_transaction[target_col] == 0][amt_col]
    amt_fraud = df_transaction[df_transaction[target_col] == 1][amt_col]

    return {
        "transaction_rows": total_transactions,
        "transaction_cols": df_transaction.shape[1],
        "identity_rows": len(df_identity) if df_identity is not None else 0,
        "identity_cols": df_identity.shape[1] if df_identity is not None else 0,
        "genuine_count": genuine_count,
        "fraud_count": fraud_count,
        "fraud_percentage": round(fraud_pct, 4),
        "imbalance_ratio": round(imbalance_ratio, 2),
        "numerical_col_count": len(num_cols),
        "categorical_col_count": len(cat_cols),
        "amount_stats": {
            "genuine_mean": float(amt_genuine.mean()) if not amt_genuine.empty else 0.0,
            "genuine_median": float(amt_genuine.median()) if not amt_genuine.empty else 0.0,
            "fraud_mean": float(amt_fraud.mean()) if not amt_fraud.empty else 0.0,
            "fraud_median": float(amt_fraud.median()) if not amt_fraud.empty else 0.0,
        },
    }


# Backward compatibility methods
EXPECTED_FEATURES = EXPECTED_FEATURES_LEGACY


def load_raw_data(filepath: Optional[Path | str] = None) -> pd.DataFrame:
    """Legacy helper for loading single raw CSV file."""
    path = Path(filepath) if filepath is not None else get_default_paths()["legacy"]
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path)
    validate_schema(df)
    return df


def validate_schema(df: pd.DataFrame) -> bool:
    """Validates dataframe schema."""
    if df.empty:
        raise ValueError("DataFrame is empty.")
    missing = [c for c in EXPECTED_FEATURES_LEGACY if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return True


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Legacy summary helper."""
    target_col = "Class" if "Class" in df.columns else "isFraud"
    amt_col = "Amount" if "Amount" in df.columns else "TransactionAmt"

    total = len(df)
    counts = df[target_col].value_counts().to_dict()
    gen = int(counts.get(0, 0))
    frd = int(counts.get(1, 0))

    amt_gen = df[df[target_col] == 0][amt_col]
    amt_frd = df[df[target_col] == 1][amt_col]

    return {
        "total_transactions": total,
        "genuine_count": gen,
        "fraud_count": frd,
        "fraud_percentage": round(frd / total * 100, 4) if total > 0 else 0.0,
        "imbalance_ratio": round(gen / frd, 2) if frd > 0 else float("inf"),
        "missing_values": int(df.isnull().sum().sum()),
        "amount_summary": {
            "genuine_mean": float(amt_gen.mean()) if not amt_gen.empty else 0.0,
            "genuine_median": float(amt_gen.median()) if not amt_gen.empty else 0.0,
            "genuine_max": float(amt_gen.max()) if not amt_gen.empty else 0.0,
            "fraud_mean": float(amt_frd.mean()) if not amt_frd.empty else 0.0,
            "fraud_median": float(amt_frd.median()) if not amt_frd.empty else 0.0,
            "fraud_max": float(amt_frd.max()) if not amt_frd.empty else 0.0,
        },
    }
