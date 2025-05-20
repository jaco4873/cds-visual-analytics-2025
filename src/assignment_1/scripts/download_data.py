"""
Script to download and extract the flower dataset for assignment 1.
"""

import os
import sys
import shutil
import urllib.request
import tarfile
from pathlib import Path
from tqdm import tqdm
from shared_lib.logger import logger
from assignment_1.config import base_config


def download_with_progress(url: str, output_path: str) -> None:
    """
    Download a file with a progress bar.

    Args:
        url: URL to download from
        output_path: Path to save the downloaded file
    """
    with urllib.request.urlopen(url) as response:
        file_size = int(response.info().get("Content-Length", 0))

        with tqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {Path(output_path).name}",
        ) as progress_bar:

            def report_progress(block_num, block_size, total_size):
                downloaded = min(block_num * block_size, total_size)
                progress_bar.update(downloaded - progress_bar.n)

            urllib.request.urlretrieve(url, output_path, reporthook=report_progress)


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
        data_dir = Path(target_dir).parent.parent
        os.makedirs(data_dir, exist_ok=True)

        temp_extract_dir = data_dir / "temp_extract"
        os.makedirs(temp_extract_dir, exist_ok=True)

        temp_file = data_dir / "17flowers.tgz"

        logger.info(f"Downloading dataset from {dataset_url}...")
        download_with_progress(dataset_url, temp_file)

        logger.info("Extracting dataset to temporary directory...")
        with tarfile.open(temp_file, "r:gz") as tar_ref:
            tar_ref.extractall(path=temp_extract_dir, filter="data")

        os.makedirs(target_dir, exist_ok=True)

        jpg_dir = temp_extract_dir / "jpg"
        if jpg_dir.exists():
            files_to_move = list(jpg_dir.glob("image_*.jpg"))
            logger.info(f"Moving {len(files_to_move)} image files to {target_dir}...")

            for file_path in tqdm(files_to_move, desc="Moving image files"):
                shutil.move(str(file_path), target_dir)

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
    DATASET_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz"

    project_root = Path(__file__).absolute().parent.parent.parent.parent
    os.chdir(project_root)

    dataset_path = project_root / base_config.dataset_folder

    if dataset_path.exists() and any(dataset_path.glob("image_*.jpg")):
        logger.info(f"Dataset already exists at {dataset_path}")
        sys.exit(0)

    success = download_and_extract_dataset(DATASET_URL, dataset_path)
    sys.exit(0 if success else 1)
