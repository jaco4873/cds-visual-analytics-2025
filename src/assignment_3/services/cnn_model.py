"""
Service for training and evaluating a CNN classifier on Lego brick images.
"""

import time
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Any

from shared_lib.logger import logger
from assignment_3.services.base_classifier_model import BaseClassifierModel


class CNNModel(BaseClassifierModel):
    """
    Service for training and evaluating a CNN classifier on Lego brick images.
    """

    def __init__(self, config):
        """
        Initialize the CNN service.

        Args:
            config: Configuration settings
        """
        super().__init__(config, model_type="cnn")

    def build_model(self, num_classes: int) -> None:
        """
        Build a CNN model for image classification.

        Args:
            num_classes: Number of classes to predict
        """
        logger.info("Building CNN model...")

        # Create a sequential model with an Input layer first
        model = models.Sequential(
            [
                # Start with an Input layer
                layers.Input(shape=self.config.cnn.input_shape),
                # Add convolutional layers
                layers.Conv2D(
                    self.config.cnn.filters[0],
                    self.config.cnn.kernel_size,
                    activation="relu",
                ),
                layers.MaxPooling2D(self.config.cnn.pool_size),
            ]
        )

        # Add more convolutional layers
        for filters in self.config.cnn.filters[1:]:
            model.add(
                layers.Conv2D(filters, self.config.cnn.kernel_size, activation="relu")
            )
            model.add(layers.MaxPooling2D(self.config.cnn.pool_size))

        # Add dense layers
        model.add(layers.Flatten())

        for units in self.config.cnn.dense_units:
            model.add(layers.Dense(units, activation="relu"))
            model.add(layers.Dropout(self.config.cnn.dropout_rate))

        # Add output layer
        model.add(layers.Dense(num_classes, activation="softmax"))

        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.cnn.learning_rate
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Print model summary
        model.summary()

        self.model = model
        logger.info("CNN model built successfully")

    def train(self, train_dataset: tf.data.Dataset, validation_dataset: tf.data.Dataset) -> dict[str, Any]:
        """
        Train the CNN model.

        Args:
            train_dataset: Training dataset
            validation_dataset: Validation dataset

        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        logger.info("Training CNN model...")
        start_time = time.time()

        # Train the model
        history = self.model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=self.config.cnn.epochs,
        )

        training_time = time.time() - start_time
        logger.info(f"Model training completed in {training_time:.2f} seconds")

        # Store the history
        self.history = history.history

        # Save the training history
        with open(self.history_path, "w") as f:
            json.dump(self.history, f)
        logger.info(f"Training history saved to {self.history_path}")

        return self.history
