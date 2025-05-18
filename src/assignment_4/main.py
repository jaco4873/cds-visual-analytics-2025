"""
Assignment 4 - Detecting Faces in Historical Newspapers

This script processes historical newspaper images to detect faces,
analyzes trends in face presence over decades, and visualizes the results.

Author: Jacob Lillelund
Date: 2024-06-05
"""

import os
import sys
import click

from assignment_4.config import config
from assignment_4.data.data_service import DataService
from assignment_4.models.face_detection_model import FaceDetectionService
from assignment_4.visualization.visualization import (
    create_newspaper_plot,
    create_comparison_plot,
    create_combined_results_csv,
    create_summary_statistics,
)

from shared_lib.logger import logger


def setup_output_dirs():
    """Set up output directories."""
    os.makedirs(config.output.base_output_dir, exist_ok=True)
    os.makedirs(config.output.results_dir, exist_ok=True)
    os.makedirs(config.output.plots_dir, exist_ok=True)
    logger.info(f"Output directories created at {config.output.base_output_dir}")


def process_newspapers(newspaper=None):
    """
    Process newspaper images to detect faces and analyze results.

    Args:
        newspaper: Optional newspaper code to process (GDL, JDG, or IMP)

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Initialize services
        data_service = DataService(config)
        face_detection_service = FaceDetectionService(config)

        # Determine which newspapers to process
        newspapers_to_process = [newspaper] if newspaper else config.data.newspapers

        # Process each newspaper
        results_dfs = []

        for newspaper in newspapers_to_process:
            logger.info(f"=== Starting processing for {newspaper} newspaper ===")

            # Process newspaper
            _, _, results_df = face_detection_service.process_newspaper(
                newspaper, data_service
            )

            # Save results
            face_detection_service.save_results(results_df, newspaper)

            # Create visualization directly using the visualization utility
            create_newspaper_plot(results_df, newspaper, config.output.plots_dir)

            # Add to list for comparison
            results_dfs.append(results_df)

            logger.info(f"=== Completed processing for {newspaper} newspaper ===")

        # Create comparison visualization if processing multiple newspapers
        if len(newspapers_to_process) > 1:
            logger.info("Creating comparison visualization...")
            create_comparison_plot(
                results_dfs, newspapers_to_process, config.output.plots_dir
            )

            # Create combined results CSV
            logger.info("Creating combined results...")
            combined_df = create_combined_results_csv(
                results_dfs, config.output.results_dir
            )

            # Create summary statistics
            logger.info("Creating summary statistics...")
            create_summary_statistics(combined_df, config.output.results_dir)

        logger.info("Processing completed successfully!")
        return 0

    except ValueError as e:
        # Configuration or input errors
        logger.error(f"Invalid input or configuration: {e}")
        return 1
    except OSError as e:
        # File system errors
        logger.error(f"File system error (check permissions): {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        return 1


@click.command()
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Path to the newspaper images directory",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    help="Path to save the output",
)
@click.option(
    "--newspaper",
    type=click.Choice(["GDL", "JDG", "IMP"], case_sensitive=True),
    help="Process only a specific newspaper",
)
def main(data_dir, output_dir, newspaper):
    """
    Detect faces in historical newspapers and analyze trends over time.

    This tool processes Swiss newspaper images (GDL, JDG, IMP) to detect human faces,
    and generates visualizations showing how the presence of faces has changed over decades.
    """
    # Update config with command line arguments if provided
    if data_dir:
        config.data.data_dir = data_dir

    if output_dir:
        config.output.base_output_dir = output_dir
        config.output.results_dir = os.path.join(output_dir, "results")
        config.output.plots_dir = os.path.join(output_dir, "plots")

    # Create output directories
    setup_output_dirs()

    # Check if data directory exists
    if not os.path.exists(config.data.data_dir):
        logger.error(f"Data directory does not exist: {config.data.data_dir}")
        return 1

    # Process newspapers
    return process_newspapers(newspaper)


if __name__ == "__main__":
    sys.exit(main())
