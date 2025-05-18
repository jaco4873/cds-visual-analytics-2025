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

from assignment_3.data.data_loader import DataLoader
from assignment_3.models.cnn_model import CNNModel
from assignment_3.models.vgg16_transfer_learning_model import (
    VGG16TransferLearningModel,
)
from assignment_3.utils.model_comparison import compare_models


def train_model(
    data_loader: DataLoader, model_type: str
) -> CNNModel | VGG16TransferLearningModel:
    """Generic function to train either model type."""
    logger.info(f"=== Starting {model_type.upper()} model training ===")

    # Load data once
    train_dataset, validation_dataset, test_dataset = data_loader.load_data()
    class_names = data_loader.get_class_names()

    # Initialize appropriate model
    if model_type == "cnn":
        model = CNNModel(config)
    elif model_type == "vgg16":
        model = VGG16TransferLearningModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Build, train and evaluate
    model.build_model(len(class_names))
    model.train(train_dataset, validation_dataset)
    model.evaluate(test_dataset)

    # Generate plots and save model
    model.plot_learning_curves()
    model.save_model()

    logger.info(f"=== {model_type.upper()} model training and evaluation completed ===")
    return model


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

    try:
        # Check if data directory exists
        if not os.path.exists(config.data.data_dir):
            logger.error(f"Data directory does not exist: {config.data.data_dir}")
            logger.error(
                "Please download the Lego dataset and place it in the correct directory."
            )
            return 1

        # Initialize the data service
        data_loader = DataLoader(config)

        if cnn_only:
            train_model(data_loader, "cnn")
        elif vgg16_only:
            train_model(data_loader, "vgg16")
        else:
            # Train both models
            train_model(data_loader, "cnn")
            train_model(data_loader, "vgg16")

            # Compare models
            logger.info("Comparing model performance...")
            compare_models(
                cnn_history_path=config.output.cnn_history_path,
                vgg16_history_path=config.output.vgg16_history_path,
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
