"""
Configuration settings for Assignment 4 - Detecting faces in historical newspapers.
"""

from pydantic_settings import BaseSettings
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


class FaceDetectionConfig(BaseSettings):
    """
    Configuration settings for the face detection model.
    """

    # MTCNN model parameters
    keep_all: bool = True  # Keep all detected faces
    min_face_size: int = 20  # Minimum face size to detect
    thresholds: list[float] = [
        0.7,
        0.9,
        0.9,
    ]  # Detection thresholds for the 3 stages
    factor: float = 0.709  # Scale factor


class DataConfig(BaseSettings):
    """
    Configuration settings for the data loading and processing.
    """

    # Path to the newspaper images directory
    data_dir: str = str(PROJECT_ROOT / "data" / "newspapers" / "images")

    # Newspapers to analyze
    newspapers: list[str] = ["GDL", "JDG", "IMP"]


class OutputConfig(BaseSettings):
    """
    Configuration settings for output files and directories.
    """

    # Base output directory
    base_output_dir: str = str(PROJECT_ROOT / "src" / "assignment_4" / "output")

    # Results directory for CSV files
    results_dir: str = f"{base_output_dir}/results"

    # Plots directory
    plots_dir: str = f"{base_output_dir}/plots"


class Config(BaseSettings):
    """
    Main configuration class that aggregates all other config classes.
    """

    face_detection: FaceDetectionConfig = FaceDetectionConfig()
    data: DataConfig = DataConfig()
    output: OutputConfig = OutputConfig()


# Create a global config instance
config = Config()
