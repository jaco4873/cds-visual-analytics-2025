"""
Logistic Regression classifier for CIFAR-10 images.
"""

import time
from sklearn.linear_model import LogisticRegression
import numpy as np

from assignment_2.classifiers.base_classifier import BaseClassifier
from assignment_2.config import CIFAR10Config
from shared_lib.utils.logger import logger


class LogisticRegressionClassifier(BaseClassifier):
    """
    CIFAR-10 image classifier using Logistic Regression.
    """

    def __init__(self, config: CIFAR10Config = None):
        """
        Initialize the Logistic Regression classifier.

        Args:
            config: Configuration object with model parameters
        """
        if config is None:
            config = CIFAR10Config(model_type="logistic_regression")
        super().__init__(config)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the Logistic Regression classifier on the training data.

        Args:
            X_train: Training features of shape (N, D) where D is the flattened image dimension
            y_train: Training labels of shape (N, 1) with class values 0-9

        Raises:
            ValueError: If input data has incorrect shape
            RuntimeError: If training fails due to convergence or other issues
        """
        lr_config = self.config.logistic_regression

        # Validate configuration
        if lr_config.max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {lr_config.max_iter}")
        if lr_config.c <= 0:
            raise ValueError(f"C must be positive, got {lr_config.c}")

        logger.info("\nTraining Logistic Regression classifier...")
        logger.info("Parameters:")
        logger.info(f"  max_iter: {lr_config.max_iter}")
        logger.info(f"  solver: {lr_config.solver}")
        logger.info(f"  C: {lr_config.c}")
        logger.info(f"  tol: {lr_config.tol}")

        try:
            # Create model
            self.model = LogisticRegression(
                max_iter=lr_config.max_iter,
                solver=lr_config.solver,
                C=lr_config.c,
                tol=lr_config.tol,
                n_jobs=-1,  # Use all available cores
                verbose=1,
            )

            # Train model with timing
            start_time = time.time()
            self.model.fit(X_train, y_train.ravel())
            training_time = time.time() - start_time

            logger.info(f"Training completed in {training_time:.2f} seconds")
            logger.info(f"Number of iterations: {self.model.n_iter_}")

            # Check for convergence issues
            if (
                hasattr(self.model, "n_iter_")
                and self.model.n_iter_[0] >= lr_config.max_iter
            ):
                logger.warning(
                    f"Logistic Regression may not have converged. Consider increasing max_iter > {lr_config.max_iter}"
                )

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise RuntimeError(f"Failed to train Logistic Regression model: {str(e)}")
