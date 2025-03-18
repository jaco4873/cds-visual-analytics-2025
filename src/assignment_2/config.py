"""
Configuration for CIFAR-10 image classification tasks.
"""

import logging
from typing import Literal, Union

from shared_lib.utils.base_config import BasePydanticConfig


class LogisticRegressionConfig(BasePydanticConfig):
    """Configuration for Logistic Regression model.

    Defines hyperparameters and training settings for logistic regression classifier.

    Attributes:
        max_iter: Maximum number of iterations for solver to converge.

        solver: Algorithm for optimization problem.
            Options:
            - 'newton-cg': Good for multinomial loss, but requires a lot of memory
            - 'lbfgs': Default optimizer, usually faster than 'newton-cg'
            - 'liblinear': Good for small datasets, only supports one-vs-rest scheme
            - 'sag': Stochastic Average Gradient, faster for large datasets
            - 'saga': Improved version of 'sag' that also handles L1 penalty

        tol: Tolerance for stopping criterion.
            Smaller values may give more accurate models but require more iterations.
            Typical range: 1e-4 to 1e-2

        c: Inverse of regularization strength.
            Smaller values specify stronger regularization (less overfitting).
            Typical range: 0.001 to 10.0
    """

    max_iter: int = 1000
    solver: str = "saga"
    tol: float = 0.001
    c: float = 0.1


class NeuralNetworkConfig(BasePydanticConfig):
    """Configuration for Neural Network model.

    Defines architecture, hyperparameters and training settings for MLP classifier.

    Attributes:
        hidden_layer_sizes: Size and number of hidden layers in the network.
            Format: tuple of integers, where length = number of hidden layers
            and each integer = number of neurons in that layer.
            Larger/more layers increase model capacity but may cause overfitting.

        activation: Activation function for hidden layers.
            Options:
            - 'relu': Rectified Linear Unit, generally performs well for most tasks
            - 'tanh': Hyperbolic tangent, range (-1, 1), can be slower
            - 'logistic': Sigmoid function, range (0, 1), can be slower
            - 'identity': No activation (linear)

        solver: Algorithm for weight optimization.
            Options:
            - 'adam': Stochastic gradient-based optimizer, works well for large datasets
            - 'sgd': Stochastic gradient descent, requires tuning learning rate
            - 'lbfgs': Optimizer in family of quasi-Newton methods, good for small datasets

        alpha: L2 regularization parameter.
            Higher values specify stronger regularization (less overfitting).
            Typical range: 1e-5 to 0.1

        batch_size: Size of minibatches for stochastic optimizers.
            Special value 'auto' uses min(200, n_samples).
            Smaller batches require more iterations but may converge faster.

        learning_rate: Schedule for weight updates.
            Options:
            - 'constant': Constant learning rate
            - 'invscaling': Gradually decreasing rate
            - 'adaptive': Keep constant if training loss continues to decrease

        max_iter: Maximum number of iterations.
            Increasing this will allow more training epochs but take longer.

        early_stopping: Whether to use early stopping to terminate training
            when validation score is not improving.

        validation_fraction: Proportion of training data to set aside for validation
            when early_stopping is True.

        n_iter_no_change: Number of iterations with no improvement to wait before
            early stopping.

        tol: Tolerance for optimization.
            Training will stop when loss improvement is less than tol.
    """

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


class CIFAR10Config(BasePydanticConfig):
    """Configuration for CIFAR-10 image classification task.

    This class defines configuration parameters for data preprocessing,
    model training, evaluation, logging, and output settings on the CIFAR-10 dataset.

    Attributes:
        run_models: Which classifier(s) to run:
            - "logistic_regression": Run only logistic regression
            - "neural_network": Run only neural network
            - "both": Run both classifiers sequentially

        grayscale: Whether to convert RGB images to grayscale.
            - True: Convert to grayscale (reduces features from 3072 to 1024)
            - False: Keep as RGB (higher dimensionality, may capture color features)

        normalize: Whether to normalize pixel values to [0, 1].
            Should generally be True for better convergence.

        logistic_regression: Configuration for LogisticRegression model.

        neural_network: Configuration for NeuralNetwork (MLPClassifier) model.

        output_dir: Directory where results will be saved.

        save_classification_report: Whether to save detailed classification metrics.

        save_loss_curve: Whether to save neural network loss curve plot.

        log_level: Logging verbosity level.
    """

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
    log_level: Union[int, str] = logging.INFO
