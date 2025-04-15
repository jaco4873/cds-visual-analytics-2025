"""
Service for training and evaluating a transfer learning classifier using VGG16.
"""

import json
import time
from typing import Any
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

from shared_lib.logger import logger
from assignment_3.services.base_classifier_model import BaseClassifierModel


class VGG16TransferLearningModel(BaseClassifierModel):
    """
    Service for training and evaluating a classifier using VGG16 as a feature extractor.
    """

    def __init__(self, config):
        """
        Initialize the transfer learning service.

        Args:
            config: Configuration settings
        """
        super().__init__(config, model_type="vgg16")

    def build_model(self, num_classes: int) -> None:
        """
        Build a model using VGG16 as a feature extractor.

        Args:
            num_classes: Number of classes to predict
        """
        logger.info("Building VGG16 transfer learning model...")

        # Load VGG16 with pre-trained weights, without the top layer
        base_model = VGG16(
            weights="imagenet",
            include_top=False,
            pooling=self.config.vgg16.pooling,
            input_shape=self.config.vgg16.input_shape,
        )

        # Freeze the convolutional layers
        for layer in base_model.layers:
            layer.trainable = False

        # Add new classifier layers using Functional API
        features = base_model.output  # The pooled features from VGG16
        x = features

        # Add all Dense layers from configuration
        for units in self.config.vgg16.dense_units:
            x = Dense(units, activation="relu")(x)
            x = BatchNormalization()(x)
            x = Dropout(self.config.vgg16.dropout_rate)(x)

        # Add output layer
        output = Dense(num_classes, activation="softmax")(x)

        # Define new model using Functional API
        model = Model(inputs=base_model.input, outputs=output)

        # Create optimizer
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=self.config.vgg16.learning_rate
        )

        # Compile the model
        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Print model summary
        model.summary()

        # Log the number of trainable parameters
        trainable_count = sum(
            [tf.keras.backend.count_params(w) for w in model.trainable_weights]
        )
        non_trainable_count = sum(
            [tf.keras.backend.count_params(w) for w in model.non_trainable_weights]
        )
        logger.info(f"Trainable parameters: {trainable_count:,}")
        logger.info(f"Non-trainable parameters: {non_trainable_count:,}")

        self.model = model
        logger.info("VGG16 transfer learning model built successfully")

    def train(self, train_dataset, validation_dataset) -> dict[str, Any]:
        """
        Train the transfer learning model.

        Args:
            train_dataset: Training dataset
            validation_dataset: Validation dataset

        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        logger.info("Training VGG16 transfer learning model...")
        start_time = time.time()

        # Train the model
        history = self.model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=self.config.vgg16.epochs,
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
