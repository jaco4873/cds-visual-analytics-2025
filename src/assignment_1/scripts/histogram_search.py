"""
Assignment 1: Image Search with Histograms

This script implements an image search algorithm using color histograms
to find similar images in a dataset of flower images.

Author: Jacob Lillelund
Date: 2025-02-28
"""

import os
import cv2
from assignment_1.services.histogram_search_service import HistogramSearchService
from shared_lib.logger import logger
from assignment_1.config import histogram_config
from assignment_1.utils.path_utils import get_dataset_path
from assignment_1.utils.result_utils import save_results_to_csv


def find_similar_images_with_histograms(
    dataset_path: str,
    target_image_path: str,
    output_path: str,
    num_results: int = 5,
    histogram_bins: tuple[int, int, int] = (8, 8, 8),
    color_space: str = "BGR",
) -> list[tuple[str, float]]:
    """
    Find images similar to a target image based on color histograms.

    Args:
        dataset_path (str): Path to the directory containing the image dataset.
        target_image_path (str): Path to the target image.
        output_path (str): Path to save the results CSV file.
        num_results (int): Number of similar images to find.
        histogram_bins (tuple): Number of bins for each channel in the histogram.
        color_space (str): Color space to use for histogram extraction ('BGR' or 'HSV').

    Returns:
        list: List of tuples containing (image_path, distance) sorted by similarity.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Initialize the histogram search service
    logger.info(f"Initializing histogram search service with dataset: {dataset_path}")
    search_service = HistogramSearchService(
        image_directory=dataset_path,
        histogram_bins=histogram_bins,
        color_space=color_space,
        comparison_method=cv2.HISTCMP_CHISQR,
    )

    # Extract histograms for all images in the dataset
    logger.info("Extracting histograms for all images...")
    search_service.extract_all_histograms()

    # Find similar images
    logger.info(f"Finding {num_results} images similar to {target_image_path}...")
    similar_images = search_service.find_similar_images(
        target_image_path=target_image_path,
        num_results=num_results,
    )

    # Save results to CSV
    logger.info(f"Saving results to {output_path}...")
    save_results_to_csv(
        results=similar_images, output_path=output_path, metric_name="Distance"
    )

    logger.info("Histogram-based image search completed successfully!")
    return similar_images


def main() -> None:
    """
    Main function to execute the histogram-based image search workflow.

    This function defines the parameters for the image search and calls
    the find_similar_images_with_histograms function.
    """

    dataset_path, target_image_path = get_dataset_path(histogram_config)

    try:
        # Find similar images using histograms
        logger.info(f"Dataset path: {dataset_path}")
        logger.info(f"Target image path: {target_image_path}")
        logger.info(f"Output path: {histogram_config.output_path}")
        logger.info(f"Number of results: {histogram_config.num_results}")
        logger.info("Starting histogram-based image search. This may take a while...")
        similar_images = find_similar_images_with_histograms(
            dataset_path=dataset_path,
            target_image_path=target_image_path,
            output_path=histogram_config.output_path,
            num_results=histogram_config.num_results,
            histogram_bins=histogram_config.histogram_bins,
            color_space=histogram_config.color_space,
        )

        logger.info("Histogram-based image search completed successfully!")

        # Log results for verification
        logger.info("Most similar images (based on histograms):")
        for i, (image_path, distance) in enumerate(similar_images):
            logger.info(f"{i + 1}. {os.path.basename(image_path)}: {distance:.4f}")

    except Exception as e:
        logger.error(f"Error during histogram-based image search: {e}")
        raise


if __name__ == "__main__":
    main()
