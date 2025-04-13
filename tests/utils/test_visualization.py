"""
Pytest tests for visualization utility functions.
"""

import os
import numpy as np
import cv2
import pytest
import matplotlib

# Set the backend to 'Agg' before importing pyplot
matplotlib.use("Agg")
from unittest.mock import patch
from shared_lib.visualization import (
    display_image,
    display_multiple_images,
    plot_histogram,
    visualize_similar_images,
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
def test_images(test_image):
    """Create a list of test images."""
    return [
        test_image.copy(),
        np.zeros((50, 50, 3), dtype=np.uint8),
        np.ones((50, 50, 3), dtype=np.uint8) * 255,
    ]


@pytest.fixture
def test_histogram():
    """Create a simple histogram."""
    return np.array([10, 20, 30, 20, 10])


@pytest.fixture
def temp_image_files(test_images):
    """Create temporary image files for testing."""
    temp_dir = "temp_test_images"
    os.makedirs(temp_dir, exist_ok=True)

    image_paths = []
    for i, img in enumerate(test_images):
        path = os.path.join(temp_dir, f"test_image_{i}.jpg")
        cv2.imwrite(path, img)
        image_paths.append(path)

    # Return paths and cleanup function
    yield image_paths, temp_dir

    # Cleanup after test
    for path in image_paths:
        if os.path.exists(path):
            os.remove(path)

    if os.path.exists(temp_dir):
        os.rmdir(temp_dir)


@pytest.fixture
def distances():
    """Create test distance values."""
    return [0.0, 0.5, 1.0]


# Tests
@patch("matplotlib.pyplot.show")
def test_display_image(mock_show, test_image):
    """Test that display_image correctly displays an image."""
    display_image(test_image, "Test Image")
    mock_show.assert_called_once()


@patch("matplotlib.pyplot.show")
def test_display_multiple_images_default(mock_show, test_images):
    """Test display_multiple_images with default parameters."""
    display_multiple_images(test_images)
    mock_show.assert_called_once()


@patch("matplotlib.pyplot.show")
def test_display_multiple_images_with_titles(mock_show, test_images):
    """Test display_multiple_images with custom titles."""
    titles = ["Image 1", "Image 2", "Image 3"]
    display_multiple_images(test_images, titles=titles)
    mock_show.assert_called_once()


@patch("matplotlib.pyplot.show")
def test_display_multiple_images_with_rows(mock_show, test_images):
    """Test display_multiple_images with custom rows."""
    display_multiple_images(test_images, rows=3, cols=1)
    mock_show.assert_called_once()


@patch("matplotlib.pyplot.show")
def test_display_multiple_images_with_cols(mock_show, test_images):
    """Test display_multiple_images with custom columns."""
    display_multiple_images(test_images, rows=1, cols=3)
    mock_show.assert_called_once()


@patch("matplotlib.pyplot.show")
def test_plot_histogram(mock_show, test_histogram):
    """Test that plot_histogram correctly plots a histogram."""
    plot_histogram(test_histogram, "Test Histogram")
    mock_show.assert_called_once()


@patch("matplotlib.pyplot.show")
def test_visualize_similar_images(mock_show, temp_image_files, distances):
    """Test that visualize_similar_images correctly visualizes similar images."""
    image_paths, _ = temp_image_files
    visualize_similar_images(image_paths, distances)
    mock_show.assert_called_once()


@patch("builtins.print")
@patch("matplotlib.pyplot.show")
def test_visualize_similar_images_with_missing_file(
    mock_show, mock_print, temp_image_files, distances
):
    """Test handling of missing image files."""
    image_paths, _ = temp_image_files
    # Add a non-existent file path
    bad_paths = image_paths + ["non_existent_file.jpg"]
    bad_distances = distances + [2.0]

    visualize_similar_images(bad_paths, bad_distances)
    mock_show.assert_called_once()
    mock_print.assert_called_once()


def test_visualize_similar_images_with_invalid_image(temp_image_files, distances):
    """Test handling of invalid image files."""
    image_paths, temp_dir = temp_image_files

    # Create an invalid image file
    invalid_path = os.path.join(temp_dir, "invalid.jpg")
    with open(invalid_path, "w") as f:
        f.write("Not an image file")

    bad_paths = image_paths + [invalid_path]
    bad_distances = distances + [2.0]

    with patch("builtins.print") as mock_print:
        visualize_similar_images(bad_paths, bad_distances)
        # Check that an error was printed
        assert mock_print.called

    # Clean up
    if os.path.exists(invalid_path):
        os.remove(invalid_path)


def test_single_image_display(test_images):
    """Test displaying a single image."""
    with patch("matplotlib.pyplot.show") as mock_show:
        display_multiple_images([test_images[0]])
        mock_show.assert_called_once()


def test_empty_image_list():
    """Test handling of an empty image list."""
    # Capture print output
    with patch("builtins.print") as mock_print:
        display_multiple_images([])
        mock_print.assert_called_once_with("No images to display")
