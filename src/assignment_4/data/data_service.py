"""
Service for loading and processing newspaper image data.
"""

import os
import re
from pathlib import Path
from PIL import Image
import pandas as pd
from collections import defaultdict

from assignment_4.config import config
from shared_lib.logger import logger


class DataService:
    """
    Service for loading and processing newspaper image data.
    """

    def __init__(self):
        """
        Initialize the data service.

        Args:
            config: Configuration settings
        """
        self.config = config
        self.data_dir = Path(config.data.data_dir)

        # Ensure the data directory exists
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory does not exist: {self.data_dir}")

        logger.info(f"Initializing data service with dataset: {self.data_dir}")

    def get_newspaper_files(self, newspaper: str) -> list[Path]:
        """
        Get all image files for a specific newspaper.

        Args:
            newspaper: Newspaper code (GDL, JDG, or IMP)

        Returns:
            List of Path objects for newspaper images
        """
        newspaper_dir = self.data_dir / newspaper

        if not newspaper_dir.exists():
            raise ValueError(f"Newspaper directory not found: {newspaper_dir}")

        return list(newspaper_dir.glob("*.jpg"))

    def extract_year_from_filename(self, filename: str) -> int:
        """
        Extract the year from a newspaper image filename.

        Args:
            filename: Name of the newspaper image file

        Returns:
            Year as integer
        """
        # Filename format example: GDL-1927-07-23-a-p0003.jpg
        pattern = r"-(\d{4})-\d{2}-\d{2}-"
        match = re.search(pattern, filename)

        if match:
            return int(match.group(1))

        raise ValueError(f"Could not extract year from filename: {filename}")

    def group_files_by_decade(self, files: list[Path]) -> dict:
        """
        Group image files by decade.

        Args:
            files: List of Path objects for newspaper images

        Returns:
            Dictionary mapping decades to lists of files
        """
        files_by_decade = defaultdict(list)

        for file_path in files:
            year = self.extract_year_from_filename(file_path.name)
            decade = (year // 10) * 10  # Convert to decade (e.g., 1923 -> 1920)
            files_by_decade[decade].append(file_path)

        return dict(files_by_decade)

    def load_image(self, image_path: Path) -> Image.Image:
        """
        Load an image from a path.

        Args:
            image_path: Path to the image file

        Returns:
            PIL Image object
        """
        try:
            return Image.open(image_path)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None

    def create_results_dataframe(
        self, newspaper: str, faces_by_decade: dict, files_by_decade: dict
    ) -> pd.DataFrame:
        """
        Create a DataFrame with face detection results by decade.

        Args:
            newspaper: Newspaper code
            faces_by_decade: Dictionary with number of faces by decade
            files_by_decade: Dictionary with files grouped by decade

        Returns:
            DataFrame with detection results
        """
        decades = sorted(files_by_decade.keys())
        data = []

        for decade in decades:
            num_pages = len(files_by_decade[decade])
            num_faces = faces_by_decade.get(decade, 0)
            pages_with_faces = sum(
                1 for count in faces_by_decade.get(f"{decade}_pages", []) if count > 0
            )
            percentage = (pages_with_faces / num_pages) * 100 if num_pages > 0 else 0

            data.append(
                {
                    "Newspaper": newspaper,
                    "Decade": f"{decade}s",
                    "Total_Pages": num_pages,
                    "Total_Faces": num_faces,
                    "Pages_With_Faces": pages_with_faces,
                    "Percentage_Pages_With_Faces": percentage,
                }
            )

        return pd.DataFrame(data)
