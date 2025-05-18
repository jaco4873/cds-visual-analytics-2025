"""
Utility for comparing model performance.
"""

import os
import json
import matplotlib.pyplot as plt

from shared_lib.logger import logger


def compare_models(
    cnn_history_path: str,
    vgg16_history_path: str,
    output_dir: str,
) -> None:
    """
    Compare performance of CNN and VGG16 models based on their training histories.
    """
    # Load histories
    cnn_history = _load_history(cnn_history_path)
    vgg16_history = _load_history(vgg16_history_path)

    # Generate visualization
    _plot_comparison(
        cnn_history, vgg16_history, os.path.join(output_dir, "model_comparison.png")
    )

    # Generate text comparison
    _generate_comparison_text(
        cnn_history, vgg16_history, os.path.join(output_dir, "model_comparison.txt")
    )


def _load_history(path: str) -> dict:
    """Load model history from JSON file."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading history from {path}: {e}")
        raise


def _plot_comparison(cnn_history: dict, vgg16_history: dict, output_path: str) -> None:
    """Generate comparison plots."""
    # Set up the figure
    plt.figure(figsize=(12, 10))

    # Plot training accuracy comparison
    plt.subplot(2, 2, 1)
    plt.plot(cnn_history["accuracy"])
    plt.plot(vgg16_history["accuracy"])
    plt.title("Training Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["CNN", "VGG16"], loc="lower right")

    # Plot validation accuracy comparison
    plt.subplot(2, 2, 2)
    plt.plot(cnn_history["val_accuracy"])
    plt.plot(vgg16_history["val_accuracy"])
    plt.title("Validation Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["CNN", "VGG16"], loc="lower right")

    # Plot training loss comparison
    plt.subplot(2, 2, 3)
    plt.plot(cnn_history["loss"])
    plt.plot(vgg16_history["loss"])
    plt.title("Training Loss Comparison")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["CNN", "VGG16"], loc="upper right")

    # Plot validation loss comparison
    plt.subplot(2, 2, 4)
    plt.plot(cnn_history["val_loss"])
    plt.plot(vgg16_history["val_loss"])
    plt.title("Validation Loss Comparison")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["CNN", "VGG16"], loc="upper right")

    plt.tight_layout()

    # Save the figure
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Model comparison plot saved to {output_path}")


def _generate_comparison_text(
    cnn_history: dict, vgg16_history: dict, output_path: str
) -> None:
    """Generate text comparison."""
    # Get final validation accuracies for comparison
    cnn_final_val_acc = cnn_history["val_accuracy"][-1]
    vgg16_final_val_acc = vgg16_history["val_accuracy"][-1]

    # Generate text comparison
    comparison_text = f"""Model Comparison:

CNN Model:
- Final validation accuracy: {cnn_final_val_acc:.4f}
- Training epochs: {len(cnn_history["accuracy"])}

VGG16 Transfer Learning Model:
- Final validation accuracy: {vgg16_final_val_acc:.4f}
- Training epochs: {len(vgg16_history["accuracy"])}

Improvement with VGG16: {(vgg16_final_val_acc - cnn_final_val_acc) * 100:.2f}%

Conclusion:
{"VGG16 transfer learning improved performance." if vgg16_final_val_acc > cnn_final_val_acc else "Direct CNN training performed better."}
"""

    # Save the comparison text
    with open(output_path, "w") as f:
        f.write(comparison_text)

    logger.info(f"Model comparison text saved to {output_path}")
