"""Exploratory Data Analysis (EDA) runner for FraudShield."""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.data.loader import load_raw_data, get_dataset_summary, get_default_data_path
from src.utils.config import get_project_root
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def generate_eda_figures(df: pd.DataFrame, output_dir: Path) -> None:
    """Generates and saves exploratory charts to the output directory.

    Args:
        df: Raw credit card transactions DataFrame.
        output_dir: Directory where figures will be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Class Distribution (Fraud vs Genuine)
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x="Class", data=df, hue="Class", palette=["#2ecc71", "#e74c3c"], legend=False)
    plt.title("Transaction Class Distribution (0: Genuine, 1: Fraud)", fontsize=12, fontweight="bold")
    plt.xlabel("Class")
    plt.ylabel("Transaction Count (Log Scale)")
    plt.yscale("log")
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{int(height):,}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=10,
                xytext=(0, 3),
                textcoords="offset points",
            )
    plt.tight_layout()
    fig_path = output_dir / "class_distribution.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {fig_path}")

    # 2. Transaction Amount Distribution by Class
    plt.figure(figsize=(8, 4))
    sns.boxplot(x="Class", y="Amount", data=df, hue="Class", palette=["#2ecc71", "#e74c3c"], showfliers=False, legend=False)
    plt.title("Transaction Amount Distribution (Excluding Extreme Outliers)", fontsize=12, fontweight="bold")
    plt.xlabel("Class (0: Genuine, 1: Fraud)")
    plt.ylabel("Transaction Amount ($)")
    plt.tight_layout()
    fig_path = output_dir / "amount_distribution_boxplot.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {fig_path}")

    # 3. Top Correlated PCA Features with Fraud (Class)
    corr = df.corr()["Class"].drop("Class").sort_values()
    top_corrs = pd.concat([corr.head(5), corr.tail(5)])
    plt.figure(figsize=(8, 5))
    top_corrs.plot(kind="barh", color=["#e74c3c" if x > 0 else "#3498db" for x in top_corrs])
    plt.title("Top Positive and Negative Correlations with Fraud (Class)", fontsize=12, fontweight="bold")
    plt.xlabel("Pearson Correlation Coefficient")
    plt.tight_layout()
    fig_path = output_dir / "top_feature_correlations.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {fig_path}")


def run_eda(filepath: Path | str | None = None) -> None:
    """Executes the full EDA report pipeline."""
    target_path = Path(filepath) if filepath else get_default_data_path()

    if not target_path.exists():
        print("\n" + "=" * 70)
        print("⚠️  DATASET NOT FOUND")
        print("=" * 70)
        print(f"Target location: {target_path}")
        print("\nPlease follow these steps to download the benchmark dataset:")
        print("1. Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("2. Download 'creditcard.csv.zip' and extract it.")
        print(f"3. Place 'creditcard.csv' into: {target_path.parent}/")
        print("4. Re-run: python -m src.data.explore")
        print("=" * 70 + "\n")
        return

    df = load_raw_data(target_path)
    summary = get_dataset_summary(df)

    print("\n" + "=" * 70)
    print(" 🛡️  FRAUDSHIELD - EXPLORATORY DATA ANALYSIS SUMMARY REPORT")
    print("=" * 70)
    print(f"Total Transactions : {summary['total_transactions']:,}")
    print(f"Genuine (Class 0)  : {summary['genuine_count']:,} ({100 - summary['fraud_percentage']:.2f}%)")
    print(f"Fraud (Class 1)    : {summary['fraud_count']:,} ({summary['fraud_percentage']:.4f}%)")
    print(f"Imbalance Ratio    : 1 fraud case for every {summary['imbalance_ratio']} genuine transactions")
    print(f"Missing Values     : {summary['missing_values']}")
    print("-" * 70)
    print("Transaction Amounts:")
    print(f"  Genuine Mean     : ${summary['amount_summary']['genuine_mean']:.2f} | Median: ${summary['amount_summary']['genuine_median']:.2f} | Max: ${summary['amount_summary']['genuine_max']:.2f}")
    print(f"  Fraud Mean       : ${summary['amount_summary']['fraud_mean']:.2f} | Median: ${summary['amount_summary']['fraud_median']:.2f} | Max: ${summary['amount_summary']['fraud_max']:.2f}")
    print("=" * 70 + "\n")

    figures_dir = get_project_root() / "reports" / "figures"
    generate_eda_figures(df, figures_dir)
    print(f"✅ Visual charts generated and saved in: {figures_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EDA on credit card dataset.")
    parser.add_argument("--file", type=str, default=None, help="Custom path to CSV file.")
    args = parser.parse_args()
    run_eda(args.file)

