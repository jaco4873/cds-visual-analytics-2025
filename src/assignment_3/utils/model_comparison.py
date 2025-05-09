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
    comparison_plot_path: str = None,
    comparison_text_path: str = None,
) -> None:
    """
    Compare performance of CNN and VGG16 models based on their training histories.

    Args:
        cnn_history_path: Path to CNN training history JSON file
        vgg16_history_path: Path to VGG16 training history JSON file
        output_dir: Directory to save the comparison outputs
        comparison_plot_path: Path to save the comparison plot (optional)
        comparison_text_path: Path to save the comparison text (optional)
    """
    logger.info("Comparing model performance...")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Set default output paths if not provided
    if comparison_plot_path is None:
        comparison_plot_path = os.path.join(output_dir, "model_comparison.png")
    if comparison_text_path is None:
        comparison_text_path = os.path.join(output_dir, "model_comparison.txt")

    # Load training histories
    try:
        with open(cnn_history_path, "r") as f:
            cnn_history = json.load(f)

        with open(vgg16_history_path, "r") as f:
            vgg16_history = json.load(f)
    except Exception as e:
        logger.error(f"Error loading model histories: {e}")
        raise

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
    plt.savefig(comparison_plot_path)
    plt.close()

    logger.info(f"Model comparison plot saved to {comparison_plot_path}")

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
    with open(comparison_text_path, "w") as f:
        f.write(comparison_text)

    logger.info(f"Model comparison text saved to {comparison_text_path}")
