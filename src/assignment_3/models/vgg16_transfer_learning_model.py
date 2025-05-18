"""
Service for training and evaluating a transfer learning classifier using VGG16.
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

from shared_lib.logger import logger
from assignment_3.models.base_classifier_model import BaseClassifierModel


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
        Build a model using VGG16 as a feature extractor with fine-tuning.

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

        # Freeze most convolutional layers, but allow fine-tuning of the last few
        trainable_layers = self.config.vgg16.trainable_layers
        if trainable_layers > 0:
            # VGG16 has 19 layers in total (including pool layers)
            # First freeze all layers
            for layer in base_model.layers:
                layer.trainable = False

            # Then unfreeze the last n layers
            for layer in base_model.layers[-trainable_layers:]:
                logger.info(f"Making layer trainable: {layer.name}")
                layer.trainable = True
        else:
            # Freeze all layers if trainable_layers is 0
            for layer in base_model.layers:
                layer.trainable = False

        # Add new classifier layers
        features = base_model.output
        x = features

        # Add all Dense layers from configuration
        for units in self.config.vgg16.dense_units:
            x = Dense(units, activation="relu")(x)
            x = BatchNormalization()(x)
            x = Dropout(self.config.vgg16.dropout_rate)(x)

        # Add output layer
        output = Dense(num_classes, activation="softmax")(x)

        # Define model
        model = Model(inputs=base_model.input, outputs=output)

        # Create optimizer with momentum
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=self.config.vgg16.learning_rate,
            momentum=self.config.vgg16.momentum,
        )

        # Compile the model
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
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
