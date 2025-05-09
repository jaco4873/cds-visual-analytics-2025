"""
Utility functions for model evaluation and results visualization.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from typing import Any

from shared_lib.logger import logger


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_file: str = None,
) -> str:
    """
    Generate a classification report and optionally save it to a file.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_file: Path to save the report (optional)

    Returns:
        The classification report as a string
    """
    # Ensure y_true is in the right shape for classification_report
    if y_true.ndim > 1 and y_true.shape[1] == 1:
        y_true = y_true.ravel()

    # Generate the report
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

    # Save to file if specified
    if output_file:
        dir_path = os.path.dirname(output_file)
        os.makedirs(dir_path, exist_ok=True)

        with open(output_file, "w") as f:
            f.write(report)

    return report


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_file: str = None,
    figsize: tuple[int, int] = (10, 8),
) -> None:
    """
    Plot a confusion matrix and optionally save to a file.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_file: Path to save the plot (optional)
        figsize: Figure size (width, height) in inches
    """
    # Ensure y_true is in the right shape
    if y_true.ndim > 1 and y_true.shape[1] == 1:
        y_true = y_true.ravel()

    # Generate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color=color)

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    # Save to file if specified
    if output_file:
        dir_path = os.path.dirname(output_file)
        os.makedirs(dir_path, exist_ok=True)
        plt.savefig(output_file)

    plt.close()


def plot_mlp_loss_curve(
    mlp: MLPClassifier, output_file: str = None, figsize: tuple[int, int] = (10, 6)
) -> None:
    """
    Plot the loss curve from a trained MLPClassifier and optionally save to a file.

    Args:
        mlp: Trained MLPClassifier instance
        output_file: Path to save the plot (optional)
        figsize: Figure size (width, height) in inches
    """
    # Check if loss curve is available
    if not hasattr(mlp, "loss_curve_"):
        logger.warning("No loss curve is available for this model")
        return

    # Plot the loss curve
    plt.figure(figsize=figsize)
    plt.plot(mlp.loss_curve_)
    plt.title("Learning Curve")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.grid(True)

    # Save to file if specified
    if output_file:
        dir_path = os.path.dirname(output_file)
        os.makedirs(dir_path, exist_ok=True)
        plt.savefig(output_file)

    plt.close()


def save_model_info(model: Any, output_file: str, include_params: bool = True) -> None:
    """
    Save model information and parameters to a text file.

    Args:
        model: The trained model
        output_file: Path to save the info
        include_params: Whether to include model parameters
    """
    dir_path = os.path.dirname(output_file)
    os.makedirs(dir_path, exist_ok=True)

    with open(output_file, "w") as f:
        # Write model type
        f.write(f"Model: {type(model).__name__}\n\n")

        if include_params:
            f.write("Parameters:\n")
            for param, value in model.get_params().items():
                f.write(f"  {param}: {value}\n")

            # Add some specific model attributes if they exist
            if hasattr(model, "n_iter_"):
                f.write(f"\nNumber of iterations: {model.n_iter_}\n")

            if hasattr(model, "n_layers_"):
                f.write(f"Number of layers: {model.n_layers_}\n")

            if hasattr(model, "n_outputs_"):
                f.write(f"Number of outputs: {model.n_outputs_}\n")

            if hasattr(model, "out_activation_"):
                f.write(f"Output activation: {model.out_activation_}\n")
