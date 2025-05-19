"""
Base service for model training and evaluation.
"""

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from typing import Any
from shared_lib.logger import logger
import json
import time
from abc import ABC, abstractmethod
from sklearn.metrics import classification_report
from assignment_3.config import Config


class BaseClassifierModel(ABC):
    """
    Base service for model training and evaluation.
    """

    def __init__(self, config: Config, model_type="base"):
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

        # Get all paths for this model type
        paths = self.config.get_model_paths(model_type)
        self.__dict__.update(paths)

        # Ensure directories exist
        self.config.ensure_model_directories(model_type)

        logger.info(f"Initializing {model_type.upper()} service")

    @abstractmethod
    def build_model(self, num_classes: int) -> None:
        """
        Build a model for image classification. To be implemented by subclasses.

        Args:
            num_classes: Number of classes to predict
        """
        pass

    def train(
        self, train_dataset: tf.data.Dataset, validation_dataset: tf.data.Dataset
    ) -> dict[str, Any]:
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

        logger.info(f"Training {self.model_type} model...")
        start_time = time.time()

        # Get config from the appropriate section based on model type
        model_config = (
            self.config.vgg16 if self.model_type == "vgg16" else self.config.cnn
        )

        # Setup callbacks
        callbacks = []

        # Add early stopping if enabled
        if model_config.early_stopping:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=model_config.early_stopping_patience,
                min_delta=model_config.early_stopping_min_delta,
                restore_best_weights=model_config.restore_best_weights,
                verbose=1,
            )
            callbacks.append(early_stopping)

        # Train the model
        history = self.model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=model_config.epochs,
            callbacks=callbacks,
        )

        training_time = time.time() - start_time
        logger.info(f"Model training completed in {training_time:.2f} seconds")

        # Store the history
        self.history = history.history

        # Save the training history
        with open(self.history_path, "w") as f:
            json.dump(self.history, f)
        logger.info(f"Training history saved to {self.history_path}")

        return self.history

    def evaluate(self, test_dataset: tf.data.Dataset) -> dict:
        """
        Evaluate the model on a separate test dataset.

        Args:
            test_dataset: Test dataset (separate from validation dataset)

        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        logger.info(f"Evaluating {self.model_type} model on test dataset...")

        # Evaluate the model on the test dataset
        test_results = self.model.evaluate(test_dataset)

        # Store results in a dictionary
        metrics = {}
        for i, name in enumerate(self.model.metrics_names):
            # Rename 'compile_metrics' to 'accuracy' for consistency
            if name == "compile_metrics":
                metrics["accuracy"] = test_results[i]
            else:
                metrics[name] = test_results[i]

        logger.info(f"Test metrics: {metrics}")

        # Save test metrics to a file
        test_metrics_path = os.path.join(
            os.path.dirname(self.report_path), "test_metrics.json"
        )
        with open(test_metrics_path, "w") as f:
            json.dump(metrics, f)
        logger.info(f"Test metrics saved to {test_metrics_path}")

        return metrics

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

    def generate_classification_report(self, test_dataset):
        """Generate and save a detailed classification report using scikit-learn."""

        # Collect all predictions and true labels
        y_true = []
        y_pred = []

        # Process the dataset batch by batch
        for x, y in test_dataset:
            # Get predictions
            predictions = self.model.predict(x, verbose=0)
            pred_classes = np.argmax(predictions, axis=1)

            # Convert to numpy arrays and append
            y_true.extend(y.numpy())
            y_pred.extend(pred_classes)

        # Generate the report
        class_names = (
            test_dataset.class_names if hasattr(test_dataset, "class_names") else None
        )
        report = classification_report(y_true, y_pred, target_names=class_names)

        # Save the report
        with open(self.report_path, "w") as f:
            f.write(report)

        logger.info(f"Classification report saved to {self.report_path}")

        return report
