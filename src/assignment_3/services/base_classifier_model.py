"""
Base service for model training and evaluation.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from typing import Any
from shared_lib.logger import logger


class BaseClassifierModel:
    """
    Base service for model training and evaluation.
    """

    def __init__(self, config, model_type="base"):
        """
        Initialize the model service.

        Args:
            config: Configuration settings
            model_type: Type of model ('cnn' or 'vgg16')
        """
        self.config = config
        self.model_type = model_type
        self.model = None
        self.history = None

        # Set output paths based on model type
        if model_type == "cnn":
            self.output_dir = self.config.output.cnn_output_dir
            self.model_path = self.config.output.cnn_model_path
            self.report_path = self.config.output.cnn_report_path
            self.history_path = self.config.output.cnn_history_path
            self.plot_path = self.config.output.cnn_plot_path
        elif model_type == "vgg16":
            self.output_dir = self.config.output.vgg16_output_dir
            self.model_path = self.config.output.vgg16_model_path
            self.report_path = self.config.output.vgg16_report_path
            self.history_path = self.config.output.vgg16_history_path
            self.plot_path = self.config.output.vgg16_plot_path
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Initializing {model_type.upper()} service")

    def build_model(self, num_classes: int) -> None:
        """
        Build a model for image classification. To be implemented by subclasses.

        Args:
            num_classes: Number of classes to predict
        """
        raise NotImplementedError("Subclasses must implement this method")

    def train(self) -> dict[str, Any]:
        """
        Train the model.

        Args:
            train_dataset: Training dataset
            validation_dataset: Validation dataset

        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        logger.info(f"Training {self.model_type.upper()} model...")

        # Each subclass should implement the specific training parameters
        return self.history

    def evaluate(
        self, test_dataset: tf.data.Dataset, class_names: list[str]
    ) -> tuple[float, float]:
        """
        Evaluate the model.

        Args:
            test_dataset: Test dataset
            class_names: List of class names

        Returns:
            Tuple of (loss, accuracy)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        logger.info(f"Evaluating {self.model_type.upper()} model...")

        # Evaluate the model
        loss, accuracy = self.model.evaluate(test_dataset)
        logger.info(f"Test loss: {loss:.4f}")
        logger.info(f"Test accuracy: {accuracy:.4f}")

        # Generate predictions
        predictions = self.model.predict(test_dataset)
        y_pred = np.argmax(predictions, axis=1)

        # Get true labels
        y_true = []
        for _, labels in test_dataset:
            y_true.extend(np.argmax(labels, axis=1))
            if len(y_true) >= len(y_pred):
                break
        y_true = y_true[: len(y_pred)]

        # Generate classification report
        report = classification_report(
            y_true, y_pred, target_names=class_names, digits=4
        )

        # Save the classification report
        with open(self.report_path, "w") as f:
            f.write(report)
        logger.info(f"Classification report saved to {self.report_path}")

        return loss, accuracy

    def plot_learning_curves(self) -> None:
        """
        Plot the learning curves following the class plot_history function.
        """
        if self.history is None:
            raise ValueError("No training history available. Train the model first.")

        logger.info("Plotting learning curves...")

        # Create the figure
        plt.figure(figsize=(12, 6))

        # Plot training & validation loss values (left subplot)
        plt.subplot(1, 2, 1)
        plt.plot(self.history["loss"], label="train_loss")
        plt.plot(self.history["val_loss"], label="val_loss", linestyle=":")
        plt.title("Loss curve")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend()
        plt.tight_layout()

        # Plot training & validation accuracy values (right subplot)
        plt.subplot(1, 2, 2)
        plt.plot(self.history["accuracy"], label="train_acc")
        plt.plot(self.history["val_accuracy"], label="val_acc", linestyle=":")
        plt.title("Accuracy curve")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend()
        plt.tight_layout()

        # Save the figure
        plt.savefig(self.plot_path)
        plt.close()

        logger.info(f"Learning curves saved to {self.plot_path}")

    def save_model(self) -> None:
        """
        Save the trained model.
        """
        if self.model is None:
            raise ValueError("No model to save. Build and train the model first.")

        logger.info(f"Saving {self.model_type.upper()} model...")

        # Save the model
        self.model.save(self.model_path)

        logger.info(f"Model saved to {self.model_path}")
