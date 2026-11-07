"""Unit tests for the data loader and validation modules."""

import pytest
import pandas as pd
import numpy as np

from src.data.loader import (
    EXPECTED_FEATURES,
    validate_schema,
    get_dataset_summary,
    load_raw_data,
)


@pytest.fixture
def mock_transaction_df():
    """Generates a small synthetic DataFrame matching the expected schema."""
    n_samples = 100
    data = {f"V{i}": np.random.randn(n_samples) for i in range(1, 29)}
    data["Time"] = np.arange(n_samples, dtype=float)
    data["Amount"] = np.random.uniform(1.0, 500.0, n_samples)
    # 95 genuine (0), 5 fraudulent (1)
    data["Class"] = [0] * 95 + [1] * 5
    return pd.DataFrame(data)


def test_validate_schema_success(mock_transaction_df):
    assert validate_schema(mock_transaction_df) is True


def test_validate_schema_empty():
    with pytest.raises(ValueError, match="DataFrame is empty"):
        validate_schema(pd.DataFrame())


def test_validate_schema_missing_column(mock_transaction_df):
    incomplete_df = mock_transaction_df.drop(columns=["V1", "Class"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_schema(incomplete_df)


def test_get_dataset_summary(mock_transaction_df):
    summary = get_dataset_summary(mock_transaction_df)
    assert summary["total_transactions"] == 100
    assert summary["genuine_count"] == 95
    assert summary["fraud_count"] == 5
    assert summary["fraud_percentage"] == 5.0
    assert summary["imbalance_ratio"] == 19.0
    assert summary["missing_values"] == 0
    assert "amount_summary" in summary
    assert summary["amount_summary"]["fraud_mean"] > 0


def test_load_raw_data_file_not_found():
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_raw_data("non_existent_file.csv")

