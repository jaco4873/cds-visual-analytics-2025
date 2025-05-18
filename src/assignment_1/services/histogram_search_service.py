"""
Service for histogram-based image search functionality.
"""

import os
import cv2
import numpy as np
from numpy import typing as npt
from assignment_1.utils.image_utils import (
    load_image,
    extract_color_histogram,
    compare_histograms,
    get_image_files,
)
from shared_lib.logger import logger
from assignment_1.utils.result_utils import save_results_to_csv as save_results


class HistogramSearchService:
    """
    Service for searching similar images based on color histograms.
    """

    def __init__(
        self,
        image_directory: str,
        histogram_bins: tuple[int, int, int] = (
            8,
            8,
            8,
        ),
        color_space: str = "BGR",
        comparison_method: int = cv2.HISTCMP_CHISQR,
    ):
        """
        Initialize the histogram search service.

        Args:
            image_directory: Directory containing the image dataset.
            histogram_bins: Number of bins for each channel in the histogram.
            color_space: Color space to use for histogram extraction ('BGR' or 'HSV').
            comparison_method: Method to use for histogram comparison.

        Raises:
            ValueError: If the image directory does not exist.
            ValueError: If the color space is not supported.
        """
        # Input validation
        if not os.path.exists(image_directory):
            raise ValueError(f"Image directory does not exist: {image_directory}")

        if color_space not in ["BGR", "HSV"]:
            raise ValueError(
                f"Unsupported color space: {color_space}. Use 'BGR' or 'HSV'."
            )

        # Initialize the histogram search service
        self.image_directory = image_directory
        self.histogram_bins = histogram_bins
        self.color_space = color_space
        self.comparison_method = comparison_method
        self.histograms: dict[str, npt.NDArray[np.float32]] = {}

        # Get all image files in the directory and fail if none are found
        try:
            self.image_files = get_image_files(image_directory)
            logger.info(f"Found {len(self.image_files)} images in {image_directory}")
        except Exception as e:
            logger.error(f"Error loading image files from {image_directory}: {e}")
            raise

    def extract_all_histograms(self) -> "HistogramSearchService":
        """
        Extract histograms for all images in the dataset.

        Returns:
            self: The service instance for method chaining.

        Raises:
            RuntimeError: If no images were found to process.
        """
        if not self.image_files:
            logger.warning("No images found to extract histograms from")
            return self

        success_count = 0
        error_count = 0

        for image_path in self.image_files:
            try:
                image = load_image(image_path)
                hist = extract_color_histogram(
                    image, bins=self.histogram_bins, color_space=self.color_space
                )
                self.histograms[image_path] = hist
                success_count += 1
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                error_count += 1

        logger.info(
            f"Extracted histograms: {success_count} successful, {error_count} failed"
        )

        if success_count == 0 and error_count > 0:
            raise RuntimeError("Failed to extract any histograms from the dataset")

        return self

    def find_similar_images(
        self, target_image_path: str, num_results: int = 5
    ) -> list[tuple[str, float]]:
        """
        Find images similar to the target image.

        Args:
            target_image_path: Path to the target image.
            num_results: Number of similar images to return (excluding the target image).

        Returns:
            List of tuples containing (image_path, distance) sorted by similarity,
            with the target image (distance 0.0) as the first element.

        Raises:
            FileNotFoundError: If the target image does not exist.
            ValueError: If num_results is less than 1.
            RuntimeError: If histogram extraction fails.
        """
        # Input validation
        if not os.path.exists(target_image_path):
            raise FileNotFoundError(f"Target image not found: {target_image_path}")

        if num_results < 1:
            raise ValueError(f"num_results must be at least 1, got {num_results}")

        try:
            # Load the target image and extract its histogram
            target_image = load_image(target_image_path)
            target_hist = extract_color_histogram(
                target_image, bins=self.histogram_bins, color_space=self.color_space
            )

            # Extract histograms if they haven't been extracted yet
            if not self.histograms:
                logger.info("No histograms found, extracting now...")
                self.extract_all_histograms()

            if not self.histograms:
                raise RuntimeError("Failed to extract histograms from the dataset")

            # Compare the target histogram with all other histograms
            results = []
            for image_path, hist in self.histograms.items():
                # Skip comparing the target image with itself
                if os.path.abspath(image_path) == os.path.abspath(target_image_path):
                    continue

                distance = compare_histograms(target_hist, hist, self.comparison_method)
                results.append((image_path, distance))

            # Sort the results based on the comparison method
            # For HISTCMP_CHISQR and HISTCMP_BHATTACHARYYA, lower values are more similar
            # For HISTCMP_CORREL and HISTCMP_INTERSECT, higher values are more similar
            if self.comparison_method in [
                cv2.HISTCMP_CHISQR,
                cv2.HISTCMP_BHATTACHARYYA,
            ]:
                results.sort(key=lambda x: x[1])
            else:
                results.sort(key=lambda x: x[1], reverse=True)

            # Add the target image at the beginning with distance 0.0
            final_results = [(target_image_path, 0.0)] + results[:num_results]

            return final_results

        except Exception as e:
            logger.error(f"Error finding similar images to {target_image_path}: {e}")
            raise

    def save_results_to_csv(
        self, results: list[tuple[str, float]], output_path: str
    ) -> None:
        """
        Save the search results to a CSV file.

        Args:
            results: List of tuples containing (image_path, distance).
            output_path: Path to save the CSV file.

        Raises:
            ValueError: If results is empty.
            IOError: If the CSV file cannot be written.
        """
        # Use "Distance" as the metric name for histogram-based results
        save_results(results, output_path, metric_name="Distance")
