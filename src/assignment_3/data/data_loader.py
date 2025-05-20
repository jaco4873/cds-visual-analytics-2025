"""
Service for loading and preprocessing Lego brick image data.
"""

import os
from typing import Literal

import tensorflow as tf

from shared_lib.logger import logger
from assignment_3.config import config


class DataLoader:
    """
    Service for loading and preprocessing Lego brick image data.
    """

    def __init__(self):
        """
        Initialize the data service.

        Args:
            config: Configuration settings
        """
        self.config = config
        self.class_names = []

        os.makedirs(config.data.data_dir, exist_ok=True)
        logger.info(f"Initializing data service with dataset: {config.data.data_dir}")

    def load_data(
        self, model_type: Literal["cnn", "vgg16"]
    ) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Load the Lego brick dataset using a proper three-way split (80/10/10).

        Args:
            model_type: Type of model ('cnn' or 'vgg16') to determine which batch size to use

        Returns:
            Tuple of (train_dataset, validation_dataset, test_dataset)
        """
        logger.info(
            f"Loading Lego dataset for {model_type} with three-way split (80/10/10)..."
        )

        img_height = self.config.data.img_height
        img_width = self.config.data.img_width

        # Select batch size based on model type
        batch_size = (
            self.config.vgg16.batch_size
            if model_type == "vgg16"
            else self.config.cnn.batch_size
        )

        # First split: 80% train, 20% temp (which will be split into validation and test)
        train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            self.config.data.data_dir,
            validation_split=self.config.data.validation_split,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size,
        )

        # Store class names from the first dataset
        self.class_names = train_dataset.class_names

        # Second split: Get the 20% temp data from the first split
        temp_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            self.config.data.data_dir,
            validation_split=self.config.data.validation_split,
            subset="validation",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size,
        )

        # Third split: Split the temp dataset into validation and test (50/50)
        temp_size = tf.data.experimental.cardinality(
            temp_dataset
        ).numpy()  # Calculate size of temp dataset to determine split
        val_size = temp_size // 2  # 50% of temp for validation

        # Create validation and test datasets
        validation_dataset = temp_dataset.take(val_size)
        test_dataset = temp_dataset.skip(val_size)

        # Enable prefetching for faster training
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

        logger.info(
            f"Created three-way split: Train={tf.data.experimental.cardinality(train_dataset).numpy()} batches, "
            f"Validation={tf.data.experimental.cardinality(validation_dataset).numpy()} batches, "
            f"Test={tf.data.experimental.cardinality(test_dataset).numpy()} batches"
        )

        return train_dataset, validation_dataset, test_dataset

    def get_class_names(self) -> list:
        """
        Get the class names from the dataset.

        Returns:
            List of class names
        """
        return self.class_names
