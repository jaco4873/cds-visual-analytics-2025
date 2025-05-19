"""
Main script for running CIFAR-10 image classification.

This script runs image classification on the CIFAR-10 dataset using
configuration-based approach.
"""

import os
import click
from assignment_2.models.logistic_regression import (
    LogisticRegressionClassifier,
)
from assignment_2.models.neural_network import NeuralNetworkClassifier
from shared_lib.logger import logger
from assignment_2.config import config


@click.command()
@click.option(
    "--model",
    type=click.Choice(
        ["logistic_regression", "neural_network", "both"], case_sensitive=False
    ),
    default="both",
    help="Which model to run",
)
def main(model: str) -> None:
    """
    Run the CIFAR-10 classification pipeline based on configuration.

    This script trains and evaluates classifiers on the CIFAR-10 dataset
    using either logistic regression, neural network, or both approaches.
    """

    # Ensure output directory exists
    os.makedirs(config.output_dir, exist_ok=True)

    # Override config with CLI parameter
    run_models = model

    logger.info("Starting CIFAR-10 classification")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Running models: {run_models}")

    # Run selected classifiers
    if run_models in ["logistic_regression", "both"]:
        logger.info("\n========== Logistic Regression Classifier ==========")
        LogisticRegressionClassifier().run_pipeline()

    if run_models in ["neural_network", "both"]:
        logger.info("\n========== Neural Network Classifier ==========")
        NeuralNetworkClassifier().run_pipeline()

    logger.info(f"\nAll outputs saved to {config.output_dir}")


if __name__ == "__main__":
    main()
