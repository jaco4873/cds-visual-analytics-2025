"""
Utility functions for image processing.
"""

import os
import cv2
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from a file path.

    Args:
        image_path: Path to the image file.

    Returns:
        The loaded image as a numpy array.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the image cannot be loaded.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    return image


def extract_color_histogram(
    image: np.ndarray, bins: list[int] = [8, 8, 8], color_space: str = "BGR"
) -> np.ndarray:
    """
    Extract a color histogram from an image.

    Args:
        image: The input image as a numpy array.
        bins: Number of bins for each channel [B, G, R] or [H, S, V].
        color_space: Color space to use ('BGR' or 'HSV').

    Returns:
        The normalized color histogram as a numpy array.
    """
    # Convert to the specified color space
    if color_space == "HSV":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the ranges for each channel
    if color_space == "HSV":
        ranges = [180, 256, 256]  # H: 0-179, S: 0-255, V: 0-255
    else:  # BGR
        ranges = [256, 256, 256]  # B, G, R: 0-255

    # Create the histogram
    hist = cv2.calcHist(
        [image], [0, 1, 2], None, bins, [0, ranges[0], 0, ranges[1], 0, ranges[2]]
    )

    # Normalize the histogram
    hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

    return hist


def compare_histograms(
    hist1: np.ndarray, hist2: np.ndarray, method: int = cv2.HISTCMP_CHISQR
) -> float:
    """
    Compare two histograms using the specified method.

    Args:
        hist1: First histogram.
        hist2: Second histogram.
        method: Comparison method (default: cv2.HISTCMP_CHISQR).
            Options include:
            - cv2.HISTCMP_CORREL: Correlation (higher is more similar)
            - cv2.HISTCMP_CHISQR: Chi-Square (lower is more similar)
            - cv2.HISTCMP_INTERSECT: Intersection (higher is more similar)
            - cv2.HISTCMP_BHATTACHARYYA: Bhattacharyya distance (lower is more similar)

    Returns:
        The comparison result as a float.
    """
    return cv2.compareHist(hist1, hist2, method)


def get_image_files(
    directory: str, extensions: list[str] = [".jpg", ".jpeg", ".png"]
) -> list[str]:
    """
    Get a list of image file paths from a directory.

    Args:
        directory: Path to the directory containing images.
        extensions: List of valid image file extensions.

    Returns:
        List of image file paths.
    """
    image_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                image_files.append(os.path.join(root, file))

    return image_files
