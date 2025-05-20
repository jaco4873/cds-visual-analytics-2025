"""
Configuration for Image Search Algorithm.
"""

from pydantic_settings import BaseSettings


class BaseSearchConfig(BaseSettings):
    """Base configuration for Image Search.

    Defines common parameters used by both search methods.

    Attributes:
        dataset_folder: Path to the directory containing the image dataset.
        target_image: Filename of the target image to find similar images for.
        output_path: Path to save the results CSV file.
        num_results: Number of similar images to find and return.
    """

    dataset_folder: str = "data/17flowers"
    target_image: str = "image_0001.jpg"
    output_path: str = "assignment_1/output/results.csv"
    num_results: int = 5


class HistogramSearchConfig(BaseSearchConfig):
    """Configuration for Histogram-based Image Search.

    Defines parameters specific to the histogram-based image search workflow.

    Attributes:
        output_path: Path to save the histogram-based results CSV file.
        histogram_bins: Number of bins for each channel in the histogram.
        color_space: Color space to use for histogram extraction.
    """

    output_path: str = "assignment_1/output/histogram_results.csv"
    histogram_bins: tuple[int, int, int] = (8, 8, 8)
    color_space: str = "BGR"


class EmbeddingSearchConfig(BaseSearchConfig):
    """Configuration for Embedding-based Image Search.

    Defines parameters specific to the embedding-based image search workflow.

    Attributes:
        output_path: Path to save the embedding-based results CSV file.
        input_shape: Input shape for the CNN model (height, width, channels).
        pooling: Pooling method for the CNN model.
        include_top: Whether to include the fully connected layers in the CNN model.
    """

    output_path: str = "assignment_1/output/embedding_results.csv"
    input_shape: tuple[int, int, int] = (224, 224, 3)
    pooling: str = "avg"
    include_top: bool = False

# Default configuration instances
base_config = BaseSearchConfig()
histogram_config = HistogramSearchConfig()
embedding_config = EmbeddingSearchConfig()
