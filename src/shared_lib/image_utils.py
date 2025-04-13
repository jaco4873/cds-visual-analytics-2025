"""
Utility functions for image processing, analysis, and preprocessing.

This module provides functions for:
- Loading and finding image files
- Image preprocessing and transformations
- Color histogram extraction and comparison
"""

import os
import cv2
import numpy as np


# ========================= Image I/O Functions =========================


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


# ========================= Image Preprocessing Functions =========================


def convert_to_grayscale(images: np.ndarray) -> np.ndarray:
    """
    Convert RGB images to grayscale.

    Args:
        images: RGB image array of shape (n_samples, height, width, 3)

    Returns:
        Grayscale image array of shape (n_samples, height, width)
    """
    # Initialize array for grayscale images
    gray_images = np.zeros(
        (images.shape[0], images.shape[1], images.shape[2]), dtype=np.uint8
    )

    # Convert each image
    for i in range(images.shape[0]):
        gray_images[i] = cv2.cvtColor(images[i], cv2.COLOR_RGB2GRAY)

    return gray_images


def normalize_images(images: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values to the range [0, 1].

    Args:
        images: Image array with values in range [0, 255]

    Returns:
        Normalized image array with values in range [0, 1]
    """
    return images.astype("float32") / 255.0


def flatten_images(images: np.ndarray) -> np.ndarray:
    """
    Flatten 2D images into 1D arrays.

    Args:
        images: Image array of shape (n_samples, height, width) or (n_samples, height, width, channels)

    Returns:
        Flattened image array of shape (n_samples, height*width) or (n_samples, height*width*channels)
    """
    n_samples = images.shape[0]
    return images.reshape(n_samples, -1)


# ========================= Image Analysis Functions =========================


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
