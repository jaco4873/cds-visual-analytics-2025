"""
Neural Network classifier (MLPClassifier) for CIFAR-10 images.
"""

import os
import numpy as np
from sklearn.neural_network import MLPClassifier

from assignment_2.models.base_classifier import BaseClassifier
from assignment_2.utils.model_evaluation import plot_mlp_loss_curve
from shared_lib.logger import logger


class NeuralNetworkClassifier(BaseClassifier):
    """
    CIFAR-10 image classifier using Neural Network (MLPClassifier).
    """

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the Neural Network classifier on the training data.

        Args:
            X_train: Training features of shape (N, D) where D is the flattened image dimension
            y_train: Training labels of shape (N, 1) with class values 0-9

        Raises:
            ValueError: If configuration parameters are invalid
            RuntimeError: If training fails
        """
        self.train_with_config(
            X_train,
            y_train,
            self.config.neural_network,
            MLPClassifier,
            extra_params={"verbose": True},
        )

    def check_convergence(self, config_section) -> None:
        """
        Check for neural network convergence issues.

        Args:
            config_section: Neural network configuration section
        """
        if self.model.n_iter_ >= config_section.max_iter:
            logger.warning(
                f"Neural network reached maximum iterations ({config_section.max_iter}). "
                "Model may not have converged. Consider increasing max_iter."
            )

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Evaluate the classifier on test data with neural network specific visualizations.

        This method extends the base evaluation by:
        1. First calling the parent class's evaluate() to perform standard evaluation
           (generating classification report, confusion matrix, etc.)
        2. Then adding neural network-specific evaluation by generating and saving
           the loss curve that visualizes the training process

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Accuracy score as a float between 0.0 and 1.0
        """
        # Get accuracy using the base implementation
        accuracy = super().evaluate(X_test, y_test)

        # Neural network specific evaluation
        if self.config.save_loss_curve and hasattr(self.model, "loss_curve_"):
            loss_curve_path = os.path.join(
                self.config.output_dir, f"{type(self).__name__}_loss_curve.png"
            )
            plot_mlp_loss_curve(self.model, output_file=loss_curve_path)
            logger.info(f"Loss curve saved to {loss_curve_path}")

        return accuracy
