"""
Main script for running CIFAR-10 image classification.

This script runs image classification on the CIFAR-10 dataset using
configuration-based approach instead of command-line arguments.
"""

import argparse
from assignment_2.config import (
    CIFAR10Config,
    LogisticRegressionConfig,
    NeuralNetworkConfig,
)
from assignment_2.classifiers.logistic_regression import (
    LogisticRegressionClassifier,
)
from assignment_2.classifiers.neural_network import NeuralNetworkClassifier
from shared_lib.utils.file_utils import ensure_directory_exists
from shared_lib.utils.logger import logger


def main() -> None:
    """Run the CIFAR-10 classification pipeline based on configuration."""

    parser = argparse.ArgumentParser(description="Run CIFAR-10 image classification")
    parser.add_argument(
        "--model",
        type=str,
        choices=["logistic_regression", "neural_network", "both"],
        default="both",
        help="Which model to run",
    )
    args = parser.parse_args()

    # Cstomize model configurations as needed or change the defaults in config:
    lr_config = LogisticRegressionConfig(
        # max_iter=2000,
        # solver="liblinear",
    )

    nn_config = NeuralNetworkConfig(
        # hidden_layer_sizes=(100,),
        # activation="relu",
        # max_iter=100,
    )

    config = CIFAR10Config(
        run_models=args.model,  # from command line argument
        grayscale=True,
        normalize=True,
        log_level="INFO",
        logistic_regression=lr_config,
        neural_network=nn_config,
    )

    # Ensure output directory exists
    ensure_directory_exists(config.output_dir)

    logger.info("Starting CIFAR-10 classification")
    logger.info(f"Output directory: {config.output_dir}")

    # Run selected classifiers
    if config.run_models in ["logistic_regression", "both"]:
        logger.info("\n========== Logistic Regression Classifier ==========")
        LogisticRegressionClassifier(config).run_pipeline()

    if config.run_models in ["neural_network", "both"]:
        logger.info("\n========== Neural Network Classifier ==========")
        NeuralNetworkClassifier(config).run_pipeline()

    logger.info(f"\nAll outputs saved to {config.output_dir}")


if __name__ == "__main__":
    main()
