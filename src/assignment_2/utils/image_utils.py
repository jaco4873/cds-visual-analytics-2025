import cv2
import numpy as np


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
