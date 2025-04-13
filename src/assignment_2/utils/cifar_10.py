import numpy as np

from tensorflow.keras.datasets import cifar10

from shared_lib.image_utils import (
    convert_to_grayscale,
    normalize_images,
    flatten_images,
)


def load_cifar10() -> tuple[
    tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]
]:
    """
    Load the CIFAR-10 dataset.

    Returns:
        ((X_train, y_train), (X_test, y_test)): Training and test data with labels
    """
    try:
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        return (X_train, y_train), (X_test, y_test)
    except ImportError:
        raise ImportError(
            "TensorFlow is required to load the CIFAR-10 dataset. "
            "Please install it using 'uv add tensorflow'."
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load CIFAR-10 dataset: {str(e)}")


def preprocess_cifar10(
    X_train: np.ndarray,
    X_test: np.ndarray,
    grayscale: bool = True,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocess CIFAR-10 images with optional grayscale conversion and normalization.

    Args:
        X_train: Training images
        X_test: Test images
        grayscale: Whether to convert images to grayscale
        normalize: Whether to normalize pixel values to [0, 1]

    Returns:
        Flattened training and test images (X_train_flat, X_test_flat)
    """
    # Validate input shapes
    if len(X_train.shape) < 3 or len(X_test.shape) < 3:
        raise ValueError(
            f"Expected 3D or 4D image arrays, got shapes {X_train.shape} and {X_test.shape}"
        )

    if grayscale:
        X_train = convert_to_grayscale(X_train)
        X_test = convert_to_grayscale(X_test)

    if normalize:
        X_train = normalize_images(X_train)
        X_test = normalize_images(X_test)

    # Flatten the images
    X_train_flat = flatten_images(X_train)
    X_test_flat = flatten_images(X_test)

    return X_train_flat, X_test_flat


def get_cifar10_class_names() -> list[str]:
    """
    Get the class names for the CIFAR-10 dataset.

    Returns:
        List of class names corresponding to labels 0-9
    """
    return [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
