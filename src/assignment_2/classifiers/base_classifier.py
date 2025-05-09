"""
Abstract base class for CIFAR-10 image classifiers.
"""

import os
from abc import ABC, abstractmethod
import numpy as np

from assignment_2.config import CIFAR10Config
from assignment_2.utils.cifar_10 import (
    load_cifar10,
    preprocess_cifar10,
    get_cifar10_class_names,
)
from assignment_2.utils.model_evaluation import (
    generate_classification_report,
    plot_confusion_matrix,
    save_model_info,
)
from shared_lib.logger import logger


class BaseClassifier(ABC):
    """Abstract base class for CIFAR-10 image classifiers.

    This class handles common functionality.

    Concrete subclasses must implement the specific
    classifier models by overriding the train() method.
    """

    def __init__(self, config: CIFAR10Config):
        """
        Initialize the classifier with configuration parameters.

        Args:
            config: Configuration object with model and preprocessing parameters
        """
        self.config = config
        self.model = None
        self.class_names = get_cifar10_class_names()

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Setup class-specific logger context
        logger.info(f"Initializing {self.__class__.__name__}")

    def load_and_preprocess_data(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess the CIFAR-10 dataset.

        Returns:
            tuple: Containing preprocessed data:
                - X_train_processed: np.ndarray of shape (50000, D) with flattened image features
                - y_train: np.ndarray of shape (50000, 1) with class labels (0-9)
                - X_test_processed: np.ndarray of shape (10000, D) with flattened image features
                - y_test: np.ndarray of shape (10000, 1) with class labels (0-9)
                where D is 1024 for grayscale or 3072 for RGB

        Raises:
            RuntimeError: If dataset loading or preprocessing fails
        """
        try:
            logger.info("Loading CIFAR-10 dataset...")
            (X_train, y_train), (X_test, y_test) = load_cifar10()

            logger.info("Original data shapes:")
            logger.info(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
            logger.info(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")

            logger.info(
                f"Preprocessing data (grayscale={self.config.grayscale}, normalize={self.config.normalize})..."
            )
            X_train_processed, X_test_processed = preprocess_cifar10(
                X_train,
                X_test,
                grayscale=self.config.grayscale,
                normalize=self.config.normalize,
            )

            logger.info("Processed data shapes:")
            logger.info(
                f"  X_train: {X_train_processed.shape}, y_train: {y_train.shape}"
            )
            logger.info(f"  X_test: {X_test_processed.shape}, y_test: {y_test.shape}")

            return X_train_processed, y_train, X_test_processed, y_test
        except Exception as e:
            logger.error(f"Error during data loading or preprocessing: {str(e)}")
            raise RuntimeError(f"Failed to load or preprocess data: {str(e)}")

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the classifier on the training data.

        This method must be implemented by all subclasses.

        Args:
            X_train: Training features
            y_train: Training labels
        """
        pass

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Evaluate the classifier on test data.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Accuracy score as a float between 0.0 and 1.0

        Raises:
            ValueError: If model hasn't been trained yet
            RuntimeError: If prediction or evaluation fails
        """
        logger.info("Evaluating model on test data...")
        if self.model is None:
            raise ValueError("Model hasn't been trained yet. Call train() first.")

        try:
            # Validate input shapes
            if len(X_test.shape) != 2:
                raise ValueError(f"Expected 2D feature array, got shape {X_test.shape}")

            # Make predictions
            y_pred = self.model.predict(X_test)

            # Generate classification report
            if self.config.save_classification_report:
                report_path = os.path.join(
                    self.config.output_dir, f"{type(self).__name__}_report.txt"
                )
                report = generate_classification_report(
                    y_test, y_pred, self.class_names, output_file=report_path
                )
                logger.info(f"Classification report saved to {report_path}")
                logger.info("\nClassification Report:")
                logger.info(report)

            # Generate confusion matrix
            cm_path = os.path.join(
                self.config.output_dir, f"{type(self).__name__}_confusion_matrix.png"
            )
            plot_confusion_matrix(y_test, y_pred, self.class_names, output_file=cm_path)
            logger.info(f"Confusion matrix saved to {cm_path}")

            # Save model info
            info_path = os.path.join(
                self.config.output_dir, f"{type(self).__name__}_info.txt"
            )
            save_model_info(self.model, info_path)
            logger.info(f"Model info saved to {info_path}")

            # Return accuracy
            accuracy = (y_pred == y_test.ravel()).mean()
            logger.info(f"Test accuracy: {accuracy:.4f}")
            return accuracy

        except Exception as e:
            logger.error(f"Error during model evaluation: {str(e)}")
            raise RuntimeError(f"Failed to evaluate model: {str(e)}")

    def run_pipeline(self) -> float:
        """
        Run the full classification pipeline: load data, train, and evaluate.

        Returns:
            Accuracy score
        """
        logger.info(f"Running full pipeline for {self.__class__.__name__}")

        # Load and preprocess data
        X_train, y_train, X_test, y_test = self.load_and_preprocess_data()

        # Train the model
        self.train(X_train, y_train)

        # Evaluate the model
        accuracy = self.evaluate(X_test, y_test)

        logger.info(f"Pipeline complete - Final accuracy: {accuracy:.4f}")
        return accuracy
