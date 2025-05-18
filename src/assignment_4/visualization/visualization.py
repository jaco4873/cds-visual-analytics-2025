"""
Utility functions for visualization of results.
"""

import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path

from shared_lib.logger import logger


def create_newspaper_plot(
    results_df: pd.DataFrame, newspaper: str, output_dir: str
) -> None:
    """
    Create a plot showing the percentage of pages with faces by decade for a single newspaper.

    Args:
        results_df: DataFrame with detection results
        newspaper: Newspaper code
        output_dir: Directory to save the plot
    """
    # Ensure plots directory exists
    plots_dir = Path(output_dir)
    os.makedirs(plots_dir, exist_ok=True)

    # Set up figure
    plt.figure(figsize=(12, 6))

    # Plot data
    plt.bar(
        results_df["Decade"],
        results_df["Percentage_Pages_With_Faces"],
        color="steelblue",
    )

    # Add labels and title
    plt.xlabel("Decade")
    plt.ylabel("Percentage of Pages with Faces")
    plt.title(f"Percentage of Pages with Faces by Decade - {newspaper}")

    # Format y-axis as percentage
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))

    # Rotate x labels for better readability
    plt.xticks(rotation=45)

    # Add grid
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Tight layout
    plt.tight_layout()

    # Save figure
    output_path = plots_dir / f"{newspaper}_faces_by_decade.png"
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Plot saved to {output_path}")


def create_comparison_plot(
    newspaper_results: list[pd.DataFrame], newspapers: list[str], output_dir: str
) -> None:
    """
    Create a comparison plot showing percentage of pages with faces across all newspapers.

    Args:
        newspaper_results: List of DataFrames with results for each newspaper
        newspapers: List of newspaper codes
        output_dir: Directory to save the plot
    """
    # Ensure the output directory exists
    plots_dir = Path(output_dir)
    os.makedirs(plots_dir, exist_ok=True)

    # Set up figure
    plt.figure(figsize=(14, 8))

    # Set up colors
    colors = ["steelblue", "firebrick", "forestgreen"]

    # Combine data for comparison
    for i, (df, newspaper) in enumerate(zip(newspaper_results, newspapers)):
        plt.plot(
            df["Decade"],
            df["Percentage_Pages_With_Faces"],
            marker="o",
            color=colors[i],
            label=newspaper,
            linewidth=2,
        )

    # Add labels and title
    plt.xlabel("Decade")
    plt.ylabel("Percentage of Pages with Faces")
    plt.title(
        "Comparison of Percentage of Pages with Faces by Decade Across Newspapers"
    )

    # Format y-axis as percentage
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))

    # Add grid
    plt.grid(linestyle="--", alpha=0.7)

    # Add legend
    plt.legend()

    # Rotate x labels for better readability
    plt.xticks(rotation=45)

    # Tight layout
    plt.tight_layout()

    # Save figure
    output_path = plots_dir / "newspaper_comparison.png"
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Comparison plot saved to {output_path}")


def create_combined_results_csv(
    newspaper_results: list[pd.DataFrame], output_dir: str
) -> pd.DataFrame:
    """
    Create a combined CSV file with results from all newspapers.

    Args:
        newspaper_results: List of DataFrames with results for each newspaper
        output_dir: Directory to save the CSV file
    """
    # Ensure the output directory exists
    results_dir = Path(output_dir)
    os.makedirs(results_dir, exist_ok=True)

    # Combine all dataframes
    combined_df = pd.concat(newspaper_results, ignore_index=True)

    # Save to CSV
    output_path = results_dir / "combined_results.csv"
    combined_df.to_csv(output_path, index=False)

    logger.info(f"Combined results saved to {output_path}")

    return combined_df


def create_summary_statistics(combined_df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """
    Create summary statistics for the detected faces.

    Args:
        combined_df: Combined DataFrame with results from all newspapers
        output_dir: Directory to save the statistics
    """
    # Ensure the output directory exists
    results_dir = Path(output_dir)
    os.makedirs(results_dir, exist_ok=True)

    # Group by newspaper to get summary statistics
    summary = combined_df.groupby("Newspaper").agg(
        {
            "Total_Pages": "sum",
            "Total_Faces": "sum",
            "Pages_With_Faces": "sum",
        }
    )

    # Calculate overall percentage of pages with faces
    summary["Overall_Percentage"] = (
        summary["Pages_With_Faces"] / summary["Total_Pages"]
    ) * 100

    # Save to CSV
    output_path = results_dir / "summary_statistics.csv"
    summary.to_csv(output_path)

    logger.info(f"Summary statistics saved to {output_path}")

    return summary
