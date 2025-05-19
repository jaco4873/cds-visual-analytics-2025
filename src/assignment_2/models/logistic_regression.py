import numpy as np
from sklearn.linear_model import LogisticRegression
from assignment_2.models.base_classifier import BaseClassifier
from shared_lib.logger import logger


class LogisticRegressionClassifier(BaseClassifier):
    """
    CIFAR-10 image classifier using Logistic Regression.
    """

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
        self.train_with_config(
            X_train,
            y_train,
            self.config.logistic_regression,
            LogisticRegression,
            extra_params={"n_jobs": -1, "verbose": 1},
        )

    def check_convergence(self, config_section) -> None:
        if (
            hasattr(self.model, "n_iter_")
            and self.model.n_iter_[0] >= config_section.max_iter
        ):
            logger.warning(
                f"Logistic Regression may not have converged. "
                f"Consider increasing max_iter > {config_section.max_iter}"
            )
