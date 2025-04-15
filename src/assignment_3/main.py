"""
Assignment 3 - Transfer Learning with Pretrained CNNs

This script implements two CNN-based approaches for Lego brick classification:
1. A CNN trained directly on the data
2. A CNN classifier using VGG16 as a feature extractor

Author: Jacob Lillelund
Date: 2024-05-15
"""

import os
import argparse
import sys


from assignment_3.config import config
from shared_lib.file_utils import ensure_directory_exists
from shared_lib.logger import logger

from assignment_3.services.data_service import DataService
from assignment_3.services.cnn_model import CNNModel
from assignment_3.services.vgg16_transfer_learning_model import (
    VGG16TransferLearningModel,
)
from assignment_3.utils.model_comparison import compare_models


def setup_output_dirs():
    """Set up output directories."""
    ensure_directory_exists(config.output.base_output_dir)
    ensure_directory_exists(config.output.cnn_output_dir)
    ensure_directory_exists(config.output.vgg16_output_dir)
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Assignment 3 - Transfer Learning with Pretrained CNNs"
    )
    parser.add_argument("--data_dir", type=str, help="Path to the Lego data directory")
    parser.add_argument("--output_dir", type=str, help="Path to save the output")
    parser.add_argument(
        "--cnn_only", action="store_true", help="Train only the CNN model"
    )
    parser.add_argument(
        "--vgg16_only", action="store_true", help="Train only the VGG16 model"
    )

    return parser.parse_args()


def main():
    """Main function to run the assignment."""
    # Parse command line arguments
    args = parse_args()

    # Update config with command line arguments if provided
    if args.data_dir:
        config.data.data_dir = args.data_dir

    if args.output_dir:
        config.output.base_output_dir = args.output_dir
        config.output.cnn_output_dir = os.path.join(args.output_dir, "cnn")
        config.output.vgg16_output_dir = os.path.join(args.output_dir, "vgg16")
        config.output.cnn_model_path = os.path.join(
            config.output.cnn_output_dir, "cnn_model.keras"
        )
        config.output.cnn_report_path = os.path.join(
            config.output.cnn_output_dir, "classification_report.txt"
        )
        config.output.cnn_history_path = os.path.join(
            config.output.cnn_output_dir, "training_history.json"
        )
        config.output.cnn_plot_path = os.path.join(
            config.output.cnn_output_dir, "learning_curves.png"
        )
        config.output.vgg16_model_path = os.path.join(
            config.output.vgg16_output_dir, "vgg16_model.keras"
        )
        config.output.vgg16_report_path = os.path.join(
            config.output.vgg16_output_dir, "classification_report.txt"
        )
        config.output.vgg16_history_path = os.path.join(
            config.output.vgg16_output_dir, "training_history.json"
        )
        config.output.vgg16_plot_path = os.path.join(
            config.output.vgg16_output_dir, "learning_curves.png"
        )
        config.output.comparison_path = os.path.join(
            args.output_dir, "model_comparison.txt"
        )

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

        if args.cnn_only:
            train_cnn_model(data_service)
        elif args.vgg16_only:
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
