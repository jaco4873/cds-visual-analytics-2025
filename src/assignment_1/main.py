"""
Assignment 1: Image Search Algorithm

This script implements a simple image search algorithm using color histograms
to find similar images in a dataset of flower images.

Author: Jacob Lillelund
Date: 2025-02-28
"""

import os
import cv2
from assignment_1.image_search_service import ImageSearchService
from shared_lib.file_utils import ensure_directory_exists
from shared_lib.logger import logger
from assignment_1.config import config


def find_similar_images(
    dataset_path: str, target_image_path: str, output_path: str, num_results: int = 5
):
    """
    Find images similar to a target image based on color histograms.

    Args:
        dataset_path (str): Path to the directory containing the image dataset.
        target_image_path (str): Path to the target image.
        output_path (str): Path to save the results CSV file.
        num_results (int): Number of similar images to find.

    Returns:
        list: List of tuples containing (image_path, distance) sorted by similarity.
    """
    # Ensure output directory exists
    ensure_directory_exists(os.path.dirname(output_path))

    # Initialize the image search service
    logger.info(f"Initializing image search service with dataset: {dataset_path}")
    search_service = ImageSearchService(
        image_directory=dataset_path,
        histogram_bins=(8, 8, 8),
        color_space="BGR",
        comparison_method=cv2.HISTCMP_CHISQR,
    )

    # Extract histograms for all images in the dataset
    logger.info("Extracting histograms for all images...")
    search_service.extract_histograms()

    # Find similar images
    logger.info(f"Finding {num_results} images similar to {target_image_path}...")
    similar_images = search_service.find_similar_images(
        target_image_path=target_image_path,
        num_results=num_results,
    )

    # Save results to CSV
    logger.info(f"Saving results to {output_path}...")
    search_service.save_results_to_csv(results=similar_images, output_path=output_path)

    logger.info("Image search completed successfully!")
    return similar_images


def main():
    """
    Main function to execute the image search workflow.

    This function defines the parameters for the image search and calls
    the find_similar_images function.
    """

    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    dataset_path = os.path.join(project_root, config.dataset_folder)
    target_image_path = os.path.join(dataset_path, config.target_image)

    try:
        # Find similar images
        logger.info(f"Dataset path: {dataset_path}")
        logger.info(f"Target image path: {target_image_path}")
        logger.info(f"Output path: {config.output_path}")
        logger.info(f"Number of results: {config.num_results}")
        logger.info("Starting image search. This may take a while...")
        similar_images = find_similar_images(
            dataset_path=dataset_path,
            target_image_path=target_image_path,
            output_path=config.output_path,
            num_results=config.num_results,
        )
        
        logger.info("Image search completed successfully!")

        # Log results for verification
        logger.info("Most similar images:")
        for i, (image_path, distance) in enumerate(similar_images):
            logger.info(f"{i + 1}. {os.path.basename(image_path)}: {distance:.4f}")

    except Exception as e:
        logger.error(f"Error during image search: {e}")
        raise


if __name__ == "__main__":
    main()
