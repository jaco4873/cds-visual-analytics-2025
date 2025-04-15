"""
Service for loading and preprocessing Lego brick image data.
"""

import os
import json
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input

from shared_lib.file_utils import ensure_directory_exists
from shared_lib.logger import logger


class DataService:
    """
    Service for loading and preprocessing Lego brick image data.
    """

    def __init__(self, config):
        """
        Initialize the data service.

        Args:
            config: Configuration settings
        """
        self.config = config
        self.class_names = []

        # Ensure the data directory exists
        if not os.path.exists(config.data.data_dir):
            raise ValueError(f"Data directory does not exist: {config.data.data_dir}")

        logger.info(f"Initializing data service with dataset: {config.data.data_dir}")

    def load_data(self) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Load the Lego brick dataset using tf.keras.preprocessing.image_dataset_from_directory.

        Returns:
            Tuple of (train_dataset, validation_dataset, test_dataset)
        """
        logger.info("Loading Lego dataset...")

        img_height = self.config.data.img_height
        img_width = self.config.data.img_width
        batch_size = self.config.cnn.batch_size

        # Load the training dataset
        train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            self.config.data.data_dir,
            validation_split=self.config.data.validation_split,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size,
        )

        validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            self.config.data.data_dir,
            validation_split=self.config.data.validation_split,
            subset="validation",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size,
        )

        # Store class names
        self.class_names = train_dataset.class_names

        # Create a test dataset (using validation for simplicity)
        test_dataset = validation_dataset

        # Apply optimization for performance
        AUTOTUNE = tf.data.AUTOTUNE

        # Function to preprocess images and convert labels to categorical format
        def preprocess(
            images: tf.Tensor, labels: tf.Tensor
        ) -> tuple[tf.Tensor, tf.Tensor]:
            # Use VGG16 preprocess_input instead of simple normalization
            images = preprocess_input(images)
            # Convert labels to categorical (one-hot encoded)
            labels = tf.one_hot(labels, depth=len(self.class_names))
            return images, labels

        # Add data augmentation to training dataset only
        def augment(
            images: tf.Tensor, labels: tf.Tensor
        ) -> tuple[tf.Tensor, tf.Tensor]:
            # Add random horizontal flipping
            images = tf.image.random_flip_left_right(images)
            return images, labels

        # Apply preprocessing to all datasets
        train_dataset = (
            train_dataset.map(preprocess)
            .cache()
            .shuffle(1000)
            .map(augment)  # Add augmentation after preprocessing
            .prefetch(buffer_size=AUTOTUNE)
        )
        validation_dataset = (
            validation_dataset.map(preprocess).cache().prefetch(buffer_size=AUTOTUNE)
        )
        test_dataset = (
            test_dataset.map(preprocess).cache().prefetch(buffer_size=AUTOTUNE)
        )

        logger.info("Dataset loading and preprocessing complete")

        return train_dataset, validation_dataset, test_dataset

    def get_class_names(self) -> list:
        """
        Get the class names from the dataset.

        Returns:
            List of class names
        """
        return self.class_names

    def save_class_names(self, output_dir: str) -> None:
        """
        Save the class names to a JSON file.

        Args:
            output_dir: Directory to save the class names
        """
        ensure_directory_exists(output_dir)
        class_names_path = os.path.join(output_dir, "class_names.json")

        with open(class_names_path, "w") as f:
            json.dump(self.class_names, f)

        logger.info(f"Class names saved to {class_names_path}")
