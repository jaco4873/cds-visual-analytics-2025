"""
Assignment 3 - Transfer Learning with Pretrained CNNs

This script implements two CNN-based approaches for Lego brick classification:
1. A CNN trained directly on the data
2. A CNN classifier using VGG16 as a feature extractor

Author: Jacob Lillelund
Date: 2024-05-15
"""

import os
import sys
import click


from assignment_3.config import config
from shared_lib.logger import logger

from assignment_3.services.data_service import DataService
from assignment_3.services.cnn_model import CNNModel
from assignment_3.services.vgg16_transfer_learning_model import (
    VGG16TransferLearningModel,
)
from assignment_3.utils.model_comparison import compare_models


def setup_output_dirs():
    """Set up output directories."""
    os.makedirs(config.output.base_output_dir, exist_ok=True)
    os.makedirs(config.output.cnn_output_dir, exist_ok=True)
    os.makedirs(config.output.vgg16_output_dir, exist_ok=True)
    logger.info(f"Output directories created at {config.output.base_output_dir}")


def train_cnn_model(data_service: DataService) -> CNNModel:
    """Train and evaluate the direct CNN model."""
    logger.info("=== Starting CNN model training ===")

    # Load data
    train_dataset, validation_dataset, test_dataset = data_service.load_data()
    class_names = data_service.get_class_names()

    # Initialize the CNN service
    cnn_service = CNNModel(config)

    # Build, train, and evaluate the model
    cnn_service.build_model(len(class_names))
    cnn_service.train(train_dataset, validation_dataset)
    cnn_service.evaluate(test_dataset, class_names)
    cnn_service.plot_learning_curves()
    cnn_service.save_model()

    logger.info("=== CNN model training and evaluation completed ===")

    return cnn_service


def train_vgg16_model(data_service: DataService) -> VGG16TransferLearningModel:
    """Train and evaluate the VGG16 transfer learning model."""
    logger.info("=== Starting VGG16 transfer learning model training ===")

    # Load data
    train_dataset, validation_dataset, test_dataset = data_service.load_data()
    class_names = data_service.get_class_names()

    # Initialize the transfer learning service
    transfer_learning_service = VGG16TransferLearningModel(config)

    # Build, train, and evaluate the model
    transfer_learning_service.build_model(len(class_names))
    transfer_learning_service.train(train_dataset, validation_dataset)
    transfer_learning_service.evaluate(test_dataset, class_names)
    transfer_learning_service.plot_learning_curves()
    transfer_learning_service.save_model()

    logger.info(
        "=== VGG16 transfer learning model training and evaluation completed ==="
    )

    return transfer_learning_service


@click.command(help="Assignment 3 - Transfer Learning with Pretrained CNNs")
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Path to the Lego data directory",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    help="Path to save the output",
)
@click.option("--cnn-only", is_flag=True, help="Train only the CNN model")
@click.option("--vgg16-only", is_flag=True, help="Train only the VGG16 model")
def main(data_dir, output_dir, cnn_only, vgg16_only):
    """Main function to run the assignment."""
    # Update config with command line arguments if provided
    if data_dir:
        config.data.data_dir = data_dir

    if output_dir:
        config.output.base_output_dir = output_dir

    # Create output directories
    setup_output_dirs()

    try:
        # Check if data directory exists
        if not os.path.exists(config.data.data_dir):
            logger.error(f"Data directory does not exist: {config.data.data_dir}")
            logger.error(
                "Please download the Lego dataset and place it in the correct directory."
            )
            return 1

        # Initialize the data service
        data_service = DataService(config)

        if cnn_only:
            train_cnn_model(data_service)
        elif vgg16_only:
            train_vgg16_model(data_service)
        else:
            # Train both models and compare them
            train_cnn_model(data_service)
            train_vgg16_model(data_service)

            logger.info("Comparing model performance...")
            compare_models(
                cnn_history_path=config.output.cnn_history_path,
                output_dir=config.output.base_output_dir,
            )

        logger.info("Assignment 3 completed successfully!")
        return 0

    except ValueError as e:
        logger.error(f"Error: {e}")
        logger.error(
            "Please ensure the Lego dataset is correctly downloaded and organized."
        )
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
