"""
Service for training and evaluating a CNN classifier on Lego brick images.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

from shared_lib.logger import logger
from assignment_3.models.base_classifier_model import BaseClassifierModel


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

        # Add convolutional layers
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
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Print model summary
        model.summary()

        self.model = model
        logger.info("CNN model built successfully")
