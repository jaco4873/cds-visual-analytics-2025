"""
Utility functions for working with search results.
"""

import os
import pandas as pd
from assignment_1.utils.image_utils import get_filename
from shared_lib.logger import logger


def save_results_to_csv(
    results: list[tuple[str, float]], output_path: str, metric_name: str = "Distance"
) -> None:
    """
    Save search results to a CSV file.

    Args:
        results: List of tuples containing (image_path, metric_value).
        output_path: Path to save the CSV file.
        metric_name: Name of the metric column in the CSV.

    Raises:
        ValueError: If results is empty.
        IOError: If the CSV file cannot be written.
    """
    # Input validation
    if not results:
        raise ValueError("Cannot save empty results")

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Create a DataFrame from the results
        df = pd.DataFrame(results, columns=["Filename", metric_name])

        # Extract the filename from the full path
        df["Filename"] = df["Filename"].apply(get_filename)

        # Save to CSV
        df.to_csv(output_path, index=False)

        logger.info(f"Results saved to {output_path}")

    except Exception as e:
        logger.error(f"Error saving results to {output_path}: {e}")
        raise IOError(f"Failed to save results to CSV: {e}")
