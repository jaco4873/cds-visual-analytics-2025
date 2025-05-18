"""Configuration for CIFAR-10 image classification tasks."""

from pydantic_settings import BaseSettings
import logging
from typing import Literal


class LogisticRegressionConfig(BaseSettings):
    """Configuration for Logistic Regression model."""

    max_iter: int = 1000
    solver: str = "saga"
    tol: float = 0.001
    c: float = 0.1


class NeuralNetworkConfig(BaseSettings):
    """Configuration for Neural Network model (MLPClassifier)."""

    hidden_layer_sizes: tuple[int, ...] = (200, 100, 50)
    activation: str = "relu"
    solver: str = "adam"
    alpha: float = 0.0001
    batch_size: int = 200
    learning_rate: str = "adaptive"
    max_iter: int = 200
    early_stopping: bool = True
    validation_fraction: float = 0.1
    n_iter_no_change: int = 10
    tol: float = 0.0001


class CIFAR10Config(BaseSettings):
    """Configuration for CIFAR-10 image classification task."""

    # Run settings
    run_models: Literal["logistic_regression", "neural_network", "both"] = "both"

    # Data preprocessing
    grayscale: bool = True
    normalize: bool = True

    # Model configurations
    logistic_regression: LogisticRegressionConfig = LogisticRegressionConfig()
    neural_network: NeuralNetworkConfig = NeuralNetworkConfig()

    # Output settings
    output_dir: str = "assignment_2/output"
    save_classification_report: bool = True
    save_loss_curve: bool = True

    # Logging settings
    log_level: int | str = logging.INFO
