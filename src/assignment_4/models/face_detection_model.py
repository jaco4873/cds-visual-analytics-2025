"""
Service for detecting faces in newspaper images using MTCNN.
"""

import os
from pathlib import Path
from PIL import Image
import torch
from facenet_pytorch import MTCNN
from collections import defaultdict
import pandas as pd
from shared_lib.logger import logger
from assignment_4.config import Config
from assignment_4.data.data_service import DataService


class FaceDetectionService:
    """
    Service for detecting faces in newspaper images using MTCNN.
    """

    def __init__(self, config: Config):
        """
        Initialize the face detection service.

        Args:
            config: Configuration settings
        """
        self.config = config

        # Initialize MTCNN model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Create MTCNN model with configuration parameters
        self.mtcnn = MTCNN(
            keep_all=config.face_detection.keep_all,
            min_face_size=config.face_detection.min_face_size,
            thresholds=config.face_detection.thresholds,
            factor=config.face_detection.factor,
            device=self.device,
        )

        logger.info("Face detection service initialized")

    def detect_faces(self, image: Image.Image) -> int:
        """
        Detect faces in an image.

        Args:
            image: PIL Image object

        Returns:
            Number of faces detected
        """
        try:
            # Detect faces
            boxes, _ = self.mtcnn.detect(image)
            return len(boxes) if boxes is not None else 0
        except RuntimeError as e:
            logger.error(f"Model error during detection: {e}")
            return 0
        except Exception as e:
            logger.error(f"Unexpected error in face detection: {e}")
            return 0

    def process_newspaper(
        self, newspaper: str, data_service: DataService
    ) -> tuple[dict, dict, pd.DataFrame]:
        """
        Process all images for a newspaper, detect faces and group by decade.

        Args:
            newspaper: Newspaper code (GDL, JDG, or IMP)
            data_service: DataService instance

        Returns:
            Tuple containing:
                - Dictionary with faces counts by decade
                - Dictionary with files grouped by decade
                - DataFrame with results
        """
        logger.info(f"Processing {newspaper} newspaper...")

        # Get all files for this newspaper
        files = data_service.get_newspaper_files(newspaper)
        logger.info(f"Found {len(files)} files for {newspaper}")

        # Group files by decade
        files_by_decade = data_service.group_files_by_decade(files)

        # Initialize counters for faces by decade
        faces_by_decade = defaultdict(int)

        # Also keep track of pages with faces for each decade
        for decade in files_by_decade:
            faces_by_decade[f"{decade}_pages"] = []

        # Process each file
        total_files = len(files)
        for i, file_path in enumerate(files):
            # Log progress every 10 files
            if (i + 1) % 10 == 0 or i == 0 or i == total_files - 1:
                logger.info(f"Processing file {i + 1}/{total_files} for {newspaper}")

            # Load image
            image = data_service.load_image(file_path)
            if image is None:
                continue

            # Detect faces
            num_faces = self.detect_faces(image)

            # Extract year and decade
            year = data_service.extract_year_from_filename(file_path.name)
            decade = (year // 10) * 10

            # Update counters
            faces_by_decade[decade] += num_faces
            faces_by_decade[f"{decade}_pages"].append(num_faces)

        # Create results DataFrame
        results_df = data_service.create_results_dataframe(
            newspaper, faces_by_decade, files_by_decade
        )

        logger.info(f"Completed processing {newspaper} newspaper")

        return faces_by_decade, files_by_decade, results_df

    def save_results(self, results_df: pd.DataFrame, newspaper: str) -> None:
        """
        Save results to CSV file.

        Args:
            results_df: DataFrame with detection results
            newspaper: Newspaper code
        """
        # Ensure output directory exists
        results_dir = Path(self.config.output.results_dir)
        os.makedirs(results_dir, exist_ok=True)

        # Save to CSV
        output_path = results_dir / f"{newspaper}_faces_by_decade.csv"
        results_df.to_csv(output_path, index=False)

        logger.info(f"Results saved to {output_path}")
