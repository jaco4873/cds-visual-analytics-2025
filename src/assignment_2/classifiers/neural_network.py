"""
Neural Network classifier (MLPClassifier) for CIFAR-10 images.
"""

import os
import time
from sklearn.neural_network import MLPClassifier
import numpy as np

from assignment_2.classifiers.base_classifier import BaseClassifier
from assignment_2.config import CIFAR10Config
from shared_lib.utils.model_evaluation import plot_mlp_loss_curve
from shared_lib.utils.logger import logger


class NeuralNetworkClassifier(BaseClassifier):
    """
    CIFAR-10 image classifier using Neural Network (MLPClassifier).
    """

    def __init__(self, config: CIFAR10Config = None):
        """
        Initialize the Neural Network classifier.

        Args:
            config: Configuration object with model parameters
        """
        if config is None:
            config = CIFAR10Config(model_type="neural_network")
        super().__init__(config)

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
        nn_config = self.config.neural_network

        # Validate configuration parameters
        if nn_config.max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {nn_config.max_iter}")
        if nn_config.alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {nn_config.alpha}")
        if nn_config.validation_fraction <= 0 or nn_config.validation_fraction >= 1:
            raise ValueError(
                f"validation_fraction must be between 0 and 1, got {nn_config.validation_fraction}"
            )

        logger.info("\nTraining Neural Network (MLPClassifier)...")
        logger.info("Parameters:")
        logger.info(f"  hidden_layer_sizes: {nn_config.hidden_layer_sizes}")
        logger.info(f"  activation: {nn_config.activation}")
        logger.info(f"  solver: {nn_config.solver}")
        logger.info(f"  alpha: {nn_config.alpha}")
        logger.info(f"  batch_size: {nn_config.batch_size}")
        logger.info(f"  learning_rate: {nn_config.learning_rate}")
        logger.info(f"  max_iter: {nn_config.max_iter}")
        logger.info(f"  early_stopping: {nn_config.early_stopping}")

        try:
            # Create model
            self.model = MLPClassifier(
                hidden_layer_sizes=nn_config.hidden_layer_sizes,
                activation=nn_config.activation,
                solver=nn_config.solver,
                alpha=nn_config.alpha,
                batch_size=nn_config.batch_size,
                learning_rate=nn_config.learning_rate,
                max_iter=nn_config.max_iter,
                early_stopping=nn_config.early_stopping,
                validation_fraction=nn_config.validation_fraction,
                n_iter_no_change=nn_config.n_iter_no_change,
                tol=nn_config.tol,
                verbose=True,
            )

            # Train model with timing
            start_time = time.time()
            self.model.fit(X_train, y_train.ravel())
            training_time = time.time() - start_time

            logger.info(f"Training completed in {training_time:.2f} seconds")
            logger.info(f"Number of iterations: {self.model.n_iter_}")

            # Warning for potential issues
            if self.model.n_iter_ >= nn_config.max_iter:
                logger.warning(
                    f"Neural network reached maximum iterations ({nn_config.max_iter}). "
                    "Model may not have converged. Consider increasing max_iter."
                )

            # Plot loss curve
            if self.config.save_loss_curve:
                loss_curve_path = os.path.join(
                    self.config.output_dir, f"{type(self).__name__}_loss_curve.png"
                )
                plot_mlp_loss_curve(self.model, output_file=loss_curve_path)
                logger.info(f"Loss curve saved to {loss_curve_path}")

        except Exception as e:
            logger.error(f"Error during neural network training: {str(e)}")
            raise RuntimeError(f"Failed to train neural network: {str(e)}")
