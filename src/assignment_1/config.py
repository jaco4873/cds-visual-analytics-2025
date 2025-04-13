"""
Configuration for Image Search Algorithm.
"""

from pydantic_settings import BaseSettings


class ImageSearchConfig(BaseSettings):
    """Configuration for Image Search Algorithm.

    Defines parameters for the image search workflow including dataset location,
    target image, output path, and number of results to return.

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


# Default configuration instance
config = ImageSearchConfig()
