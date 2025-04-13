"""
Utility functions for visualization.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def display_image(
    image: np.ndarray, title: str = "Image", figsize: tuple[int, int] = (8, 8), **kwargs
) -> None:
    """
    Display an image using matplotlib.

    Args:
        image: The image to display (BGR format).
        title: Title for the plot.
        figsize: Figure size (width, height) in inches.
        **kwargs: Additional keyword arguments to pass to plt.imshow().
    """
    # Convert BGR to RGB for matplotlib
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=figsize)
    plt.imshow(rgb_image, **kwargs)
    plt.title(title)
    plt.axis("off")
    plt.show()


def display_multiple_images(
    images: list[np.ndarray],
    titles: list[str] = None,
    figsize: tuple[int, int] = (15, 10),
    rows: int = None,
    cols: int = None,
    **kwargs,
) -> None:
    """
    Display multiple images in a grid.

    Args:
        images: List of images to display (BGR format).
        titles: List of titles for each image.
        figsize: Figure size (width, height) in inches.
        rows: Number of rows in the grid. If None, calculated automatically.
        cols: Number of columns in the grid. If None, calculated automatically.
        **kwargs: Additional keyword arguments to pass to plt.imshow().
    """
    n = len(images)

    # Handle empty image list
    if n == 0:
        print("No images to display")
        return  # Early return, don't try to show anything

    if titles is None:
        titles = [f"Image {i + 1}" for i in range(n)]

    if rows is None and cols is None:
        # Calculate a reasonable grid size
        cols = min(5, n)
        rows = (n + cols - 1) // cols
    elif rows is None:
        rows = (n + cols - 1) // cols
    elif cols is None:
        cols = (n + rows - 1) // rows

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Handle the case where there's only one row or column
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    elif rows == 1 or cols == 1:
        axes = axes.flatten()

    for i in range(rows * cols):
        if i < n:
            # Convert BGR to RGB for matplotlib
            rgb_image = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)

            if rows == 1 and cols == 1:
                axes[0].imshow(rgb_image, **kwargs)
                axes[0].set_title(titles[i])
                axes[0].axis("off")
            else:
                row, col = i // cols, i % cols
                if rows == 1:
                    axes[col].imshow(rgb_image, **kwargs)
                    axes[col].set_title(titles[i])
                    axes[col].axis("off")
                elif cols == 1:
                    axes[row].imshow(rgb_image, **kwargs)
                    axes[row].set_title(titles[i])
                    axes[row].axis("off")
                else:
                    axes[row, col].imshow(rgb_image, **kwargs)
                    axes[row, col].set_title(titles[i])
                    axes[row, col].axis("off")
        else:
            # Hide empty subplots
            row, col = i // cols, i % cols
            if rows == 1:
                axes[col].axis("off")
            elif cols == 1:
                axes[row].axis("off")
            else:
                axes[row, col].axis("off")

    plt.tight_layout()
    plt.show()


def plot_histogram(
    hist: np.ndarray, title: str = "Histogram", figsize: tuple[int, int] = (10, 6)
) -> None:
    """
    Plot a 1D histogram.

    Args:
        hist: The histogram to plot.
        title: Title for the plot.
        figsize: Figure size (width, height) in inches.
    """
    plt.figure(figsize=figsize)
    plt.plot(hist)
    plt.title(title)
    plt.xlabel("Bin")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.show()


def visualize_similar_images(
    image_paths: list[str], distances: list[float], figsize: tuple[int, int] = (15, 10)
) -> None:
    """
    Visualize a set of similar images with their distance metrics.

    Args:
        image_paths: List of paths to the images.
        distances: List of distance metrics corresponding to each image.
        figsize: Figure size (width, height) in inches.
    """
    images = []
    titles = []

    for i, (path, distance) in enumerate(zip(image_paths, distances)):
        try:
            image = cv2.imread(path)
            if image is None:
                print(f"Warning: Could not load image {path}")
                continue

            images.append(image)

            # Create a title with the filename and distance
            filename = os.path.basename(path)
            if i == 0:
                titles.append(f"{filename}\nTarget Image")
            else:
                titles.append(f"{filename}\nDistance: {distance:.4f}")
        except Exception as e:
            print(f"Error processing {path}: {e}")

    display_multiple_images(images, titles, figsize=figsize)
