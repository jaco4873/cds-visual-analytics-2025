"""
Script to download and extract the flower dataset for assignment 1.
"""

import os
import sys
import shutil
import urllib.request
import tarfile
from pathlib import Path
from assignment_1.config import config
from shared_lib.logger import logger


def download_and_extract_dataset(dataset_url: str, target_dir: str) -> bool:
    """
    Download and extract the flower dataset.

    Args:
        dataset_url: URL to download the dataset from
        target_dir: Directory where the dataset should be extracted

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create data directory if it doesn't exist
        data_dir = Path(target_dir).parent
        os.makedirs(data_dir, exist_ok=True)

        # Create a temporary extraction directory
        temp_extract_dir = data_dir / "temp_extract"
        os.makedirs(temp_extract_dir, exist_ok=True)

        # Temporary tgz file path
        temp_file = data_dir / "17flowers.tgz"

        logger.info(f"Downloading dataset from {dataset_url}...")
        urllib.request.urlretrieve(dataset_url, temp_file)

        logger.info("Extracting dataset to temporary directory...")
        with tarfile.open(temp_file, "r:gz") as tar_ref:
            tar_ref.extractall(path=temp_extract_dir, filter="data")

        # Create the target directory if it doesn't exist
        os.makedirs(target_dir, exist_ok=True)

        # Move all jpg files to the target directory
        jpg_dir = temp_extract_dir / "jpg"
        if jpg_dir.exists():
            logger.info(f"Moving image files to {target_dir}...")
            # Find all jpg files
            for file_path in jpg_dir.glob("image_*.jpg"):
                # Move each jpg file to the target directory
                shutil.move(str(file_path), target_dir)

        # Clean up temporary files and directories
        if temp_file.exists():
            os.remove(temp_file)
        if temp_extract_dir.exists():
            shutil.rmtree(temp_extract_dir)

        logger.info("Dataset download and extraction complete!")
        return True

    except Exception as e:
        logger.error(f"Error downloading or extracting dataset: {e}")
        return False


if __name__ == "__main__":
    # URL for the 17 Category Flower Dataset
    DATASET_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz"

    # Get the project root directory (2 levels up from this script)
    project_root = Path(__file__).absolute().parent.parent.parent

    # Change to project root directory to ensure paths are correct
    os.chdir(project_root)

    # Check if dataset already exists
    if os.path.exists(config.dataset_folder) and any(
        Path(config.dataset_folder).glob("image_*.jpg")
    ):
        logger.info(f"Dataset already exists at {config.dataset_folder}")
        sys.exit(0)

    # Download and extract dataset
    success = download_and_extract_dataset(DATASET_URL, config.dataset_folder)
    sys.exit(0 if success else 1)
