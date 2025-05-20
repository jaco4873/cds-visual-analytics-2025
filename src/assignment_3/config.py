"""
Configuration settings for Assignment 3 using Pydantic BaseSettings.
"""

from pydantic_settings import BaseSettings
from pathlib import Path
import os

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


class CNNConfig(BaseSettings):
    """
    Configuration settings for the CNN model.
    """

    # Input shape for the model (height, width, channels)
    input_shape: tuple[int, int, int] = (224, 224, 3)

    # Training parameters
    batch_size: int = 32
    epochs: int = 30

    # Model parameters
    learning_rate: float = 0.001

    # CNN architecture parameters
    filters: list[int] = [32, 64, 128]
    kernel_size: tuple[int, int] = (3, 3)
    pool_size: tuple[int, int] = (2, 2)
    dense_units: list[int] = [512, 256]
    dropout_rate: float = 0.5

    # Early stopping parameters
    early_stopping: bool = True
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 0.001
    restore_best_weights: bool = True


class VGG16Config(BaseSettings):
    """
    Configuration settings for the VGG16 transfer learning model.
    """

    # Input shape for the model (must match VGG16 expectations)
    input_shape: tuple[int, int, int] = (224, 224, 3)

    # Training parameters
    batch_size: int = 32
    epochs: int = 30

    # Transfer learning parameters
    include_top: bool = False
    pooling: str = "avg"

    # Fine-tuning parameters
    learning_rate: float = 0.0005
    momentum: float = 0.9
    trainable_layers: int = 4  # Unfreeze last 4 layers of VGG16

    # Top layers configuration
    dense_units: list[int] = [256, 128]  # Two layers with 256 and 128 units
    dropout_rate: float = 0.5

    # Early stopping parameters
    early_stopping: bool = True
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.01
    restore_best_weights: bool = True


class DataConfig(BaseSettings):
    """
    Configuration settings for the data loading and processing.
    """

    # Path to the dataset (will be defined in main script)
    data_dir: str = str(PROJECT_ROOT / "data" / "lego" / "cropped")

    # Image settings
    img_height: int = 224
    img_width: int = 224

    # Data splitting
    validation_split: float = 0.2


class OutputConfig(BaseSettings):
    """
    Configuration settings for output files and directories.
    """

    # Store the base directory
    base_output_dir: str = str(PROJECT_ROOT / "src" / "assignment_3" / "output")

    # Derive output directories from base_output_dir
    @property
    def cnn_output_dir(self) -> str:
        return os.path.join(self.base_output_dir, "cnn")

    @property
    def vgg16_output_dir(self) -> str:
        return os.path.join(self.base_output_dir, "vgg16")

    @property
    def cnn_model_path(self) -> str:
        return os.path.join(self.cnn_output_dir, "cnn_model.keras")

    @property
    def cnn_report_path(self) -> str:
        return os.path.join(self.cnn_output_dir, "classification_report.txt")

    @property
    def cnn_history_path(self) -> str:
        return os.path.join(self.cnn_output_dir, "training_history.json")

    @property
    def cnn_plot_path(self) -> str:
        return os.path.join(self.cnn_output_dir, "learning_curves.png")

    @property
    def vgg16_model_path(self) -> str:
        return os.path.join(self.vgg16_output_dir, "vgg16_model.keras")

    @property
    def vgg16_report_path(self) -> str:
        return os.path.join(self.vgg16_output_dir, "classification_report.txt")

    @property
    def vgg16_history_path(self) -> str:
        return os.path.join(self.vgg16_output_dir, "training_history.json")

    @property
    def vgg16_plot_path(self) -> str:
        return os.path.join(self.vgg16_output_dir, "learning_curves.png")

    @property
    def comparison_path(self) -> str:
        return os.path.join(self.base_output_dir, "model_comparison.txt")


class Config(BaseSettings):
    """
    Main configuration class that aggregates all other config classes.
    """

    data: DataConfig = DataConfig()
    cnn: CNNConfig = CNNConfig()
    vgg16: VGG16Config = VGG16Config()
    output: OutputConfig = OutputConfig()

    def get_model_paths(self, model_type: str) -> dict:
        """
        Get all paths for a specific model type.

        Args:
            model_type: Type of model ('cnn' or 'vgg16')

        Returns:
            Dictionary containing all paths for the model
        """
        if model_type == "cnn":
            return {
                "output_dir": self.output.cnn_output_dir,
                "model_path": self.output.cnn_model_path,
                "report_path": self.output.cnn_report_path,
                "history_path": self.output.cnn_history_path,
                "plot_path": self.output.cnn_plot_path,
            }
        elif model_type == "vgg16":
            return {
                "output_dir": self.output.vgg16_output_dir,
                "model_path": self.output.vgg16_model_path,
                "report_path": self.output.vgg16_report_path,
                "history_path": self.output.vgg16_history_path,
                "plot_path": self.output.vgg16_plot_path,
            }
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def ensure_model_directories(self, model_type: str) -> None:
        """Create all necessary directories for a model."""
        paths = self.get_model_paths(model_type)
        os.makedirs(paths["output_dir"], exist_ok=True)


# Create a global config instance
config = Config()
