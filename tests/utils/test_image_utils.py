"""
Pytest tests for image utility functions.
"""

import os
import numpy as np
import cv2
import pytest
from shared_lib.image_utils import (
    load_image,
    extract_color_histogram,
    compare_histograms,
    get_image_files,
)


# Fixtures
@pytest.fixture
def test_image():
    """Create a simple test image (50x50 RGB)."""
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    image[:25, :25] = [255, 0, 0]  # Red square
    image[25:, 25:] = [0, 255, 0]  # Green square
    return image


@pytest.fixture
def temp_image_file(test_image):
    """Create a temporary image file for testing."""
    temp_dir = "temp_test_images"
    os.makedirs(temp_dir, exist_ok=True)

    image_path = os.path.join(temp_dir, "test_image.jpg")
    cv2.imwrite(image_path, test_image)

    # Return path and cleanup function
    yield image_path

    # Cleanup after test
    if os.path.exists(image_path):
        os.remove(image_path)

    if os.path.exists(temp_dir):
        os.rmdir(temp_dir)


@pytest.fixture
def temp_image_directory():
    """Create a temporary directory with image files for testing."""
    temp_dir = "temp_test_images_dir"
    os.makedirs(temp_dir, exist_ok=True)

    # Create some test images with different extensions
    image_paths = []
    for ext in [".jpg", ".png", ".txt", ".jpeg"]:
        path = os.path.join(temp_dir, f"test_image{ext}")
        # For non-image files, just create an empty file
        if ext == ".txt":
            with open(path, "w") as f:
                f.write("Not an image")
        else:
            # Create a simple image
            img = np.zeros((10, 10, 3), dtype=np.uint8)
            cv2.imwrite(path, img)
        image_paths.append(path)

    # Return directory path and cleanup function
    yield temp_dir

    # Cleanup after test
    for path in image_paths:
        if os.path.exists(path):
            os.remove(path)

    if os.path.exists(temp_dir):
        os.rmdir(temp_dir)


# Tests for load_image
def test_load_image_success(temp_image_file):
    """Test that load_image correctly loads an existing image."""
    image = load_image(temp_image_file)
    assert isinstance(image, np.ndarray)
    assert image.shape[2] == 3  # Should be a color image


def test_load_image_file_not_found():
    """Test that load_image raises FileNotFoundError for non-existent files."""
    with pytest.raises(FileNotFoundError):
        load_image("non_existent_file.jpg")


def test_load_image_invalid_image():
    """Test that load_image raises ValueError for invalid image files."""
    # Create a temporary text file
    with open("temp_invalid.txt", "w") as f:
        f.write("Not an image file")

    try:
        with pytest.raises(ValueError):
            load_image("temp_invalid.txt")
    finally:
        # Clean up
        if os.path.exists("temp_invalid.txt"):
            os.remove("temp_invalid.txt")


# Tests for extract_color_histogram
def test_extract_color_histogram_bgr(test_image):
    """Test extracting color histogram in BGR color space."""
    hist = extract_color_histogram(test_image, bins=[8, 8, 8], color_space="BGR")
    assert isinstance(hist, np.ndarray)
    assert hist.max() <= 1.0  # Should be normalized
    assert hist.min() >= 0.0


def test_extract_color_histogram_hsv(test_image):
    """Test extracting color histogram in HSV color space."""
    hist = extract_color_histogram(test_image, bins=[8, 8, 8], color_space="HSV")
    assert isinstance(hist, np.ndarray)
    assert hist.max() <= 1.0  # Should be normalized
    assert hist.min() >= 0.0


def test_extract_color_histogram_custom_bins(test_image):
    """Test extracting color histogram with custom bin sizes."""
    hist = extract_color_histogram(test_image, bins=[4, 4, 4], color_space="BGR")
    assert isinstance(hist, np.ndarray)
    assert hist.shape == (4, 4, 4)


# Tests for compare_histograms
def test_compare_histograms_identical():
    """Test comparing identical histograms."""
    # Create a proper histogram format that OpenCV expects
    # OpenCV histograms need to be 1D or have specific shape and be float32
    hist = np.ones((8, 8, 8), dtype=np.float32)
    hist = hist.reshape(-1)  # Flatten to 1D

    # For identical histograms, HISTCMP_CHISQR should return 0
    result = compare_histograms(hist, hist, method=cv2.HISTCMP_CHISQR)
    assert result == 0.0


def test_compare_histograms_different():
    """Test comparing different histograms."""
    hist1 = np.ones((8, 8, 8), dtype=np.float32)
    hist2 = np.zeros((8, 8, 8), dtype=np.float32)

    # Reshape to 1D arrays
    hist1 = hist1.reshape(-1)
    hist2 = hist2.reshape(-1)

    # For completely different histograms, HISTCMP_CHISQR should return a positive value
    result = compare_histograms(hist1, hist2, method=cv2.HISTCMP_CHISQR)
    assert result > 0.0


def test_compare_histograms_different_methods():
    """Test comparing histograms with different methods."""
    hist1 = np.ones((8, 8, 8), dtype=np.float32)
    hist2 = np.ones((8, 8, 8), dtype=np.float32) * 0.5

    # Reshape to 1D arrays
    hist1 = hist1.reshape(-1)
    hist2 = hist2.reshape(-1)

    # Test different comparison methods
    methods = [
        cv2.HISTCMP_CORREL,
        cv2.HISTCMP_CHISQR,
        cv2.HISTCMP_INTERSECT,
        cv2.HISTCMP_BHATTACHARYYA,
    ]

    for method in methods:
        result = compare_histograms(hist1, hist2, method=method)
        assert isinstance(result, float)


# Tests for get_image_files
def test_get_image_files(temp_image_directory):
    """Test getting image files from a directory."""
    image_files = get_image_files(temp_image_directory)

    # Should find 3 image files (.jpg, .jpeg, .png) and not the .txt file
    assert len(image_files) == 3

    # Check that all returned files have valid extensions
    for file in image_files:
        ext = os.path.splitext(file)[1].lower()
        assert ext in [".jpg", ".jpeg", ".png"]


def test_get_image_files_custom_extensions(temp_image_directory):
    """Test getting image files with custom extensions."""
    # Only look for .png files
    image_files = get_image_files(temp_image_directory, extensions=[".png"])

    assert len(image_files) == 1
    assert all(file.endswith(".png") for file in image_files)


def test_get_image_files_empty_directory():
    """Test getting image files from an empty directory."""
    # Create a temporary empty directory
    temp_dir = "temp_empty_dir"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        image_files = get_image_files(temp_dir)
        assert len(image_files) == 0
    finally:
        # Clean up
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)


def test_get_image_files_nonexistent_directory():
    """Test getting image files from a non-existent directory."""
    image_files = get_image_files("non_existent_directory")
    assert len(image_files) == 0
