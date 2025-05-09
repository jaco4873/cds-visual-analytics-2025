"""
Service for image search functionality using deep learning embeddings.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

from assignment_1.utils.image_utils import get_image_files, get_filename
from shared_lib.logger import logger


class EmbeddingSearchService:
    """
    Service for searching similar images based on deep learning embeddings.
    """

    def __init__(
        self,
        image_directory: str,
        input_shape: tuple[int, int, int] = (224, 224, 3),
        pooling: str = "avg",
        include_top: bool = False,
    ):
        """
        Initialize the embedding search service.

        Args:
            image_directory: Directory containing the image dataset.
            input_shape: Input shape for the model (height, width, channels).
            pooling: Pooling strategy for the model ('avg' or 'max').
            include_top: Whether to include the top (fully connected) layers.

        Raises:
            ValueError: If the image directory does not exist.
            ValueError: If pooling is not supported.
        """
        # Input validation
        if not os.path.exists(image_directory):
            raise ValueError(f"Image directory does not exist: {image_directory}")

        if pooling not in ["avg", "max", None]:
            raise ValueError(
                f"Unsupported pooling: {pooling}. Use 'avg', 'max', or None."
            )

        # Initialize the embedding search service
        self.image_directory = image_directory
        self.input_shape = input_shape
        self.embeddings: dict[str, np.ndarray] = {}

        # Load the VGG16 model
        logger.info("Loading VGG16 model...")
        self.model = VGG16(
            weights="imagenet",
            include_top=include_top,
            pooling=pooling,
            input_shape=input_shape,
        )
        logger.info("VGG16 model loaded successfully")

        # Get all image files in the directory and fail if none are found
        try:
            self.image_files = get_image_files(image_directory)
            logger.info(f"Found {len(self.image_files)} images in {image_directory}")
        except Exception as e:
            logger.error(f"Error loading image files from {image_directory}: {e}")
            raise

    def extract_features(self, image_path: str) -> np.ndarray:
        """
        Extract features from an image using the VGG16 model.

        Args:
            image_path: Path to the image.

        Returns:
            Feature vector for the image.

        Raises:
            FileNotFoundError: If the image does not exist.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            # Load and preprocess the image
            img = load_img(
                image_path, target_size=(self.input_shape[0], self.input_shape[1])
            )
            img_array = img_to_array(img)
            # Expand dimensions to match the input shape expected by the model
            expanded_img = np.expand_dims(img_array, axis=0)
            # Preprocess the image for VGG16
            preprocessed_img = preprocess_input(expanded_img)

            # Extract features using the model
            features = self.model.predict(preprocessed_img, verbose=0)
            # Flatten the features
            flattened_features = features.flatten()

            return flattened_features

        except Exception as e:
            logger.error(f"Error extracting features from {image_path}: {e}")
            raise

    def extract_all_embeddings(self) -> None:
        """
        Extract embeddings for all images in the dataset.

        Raises:
            RuntimeError: If no images were found to process.
        """
        if not self.image_files:
            logger.warning("No images found to extract embeddings from")
            return

        success_count = 0
        error_count = 0

        logger.info(f"Extracting embeddings for {len(self.image_files)} images...")

        for image_path in self.image_files:
            try:
                embedding = self.extract_features(image_path)
                self.embeddings[image_path] = embedding
                success_count += 1

                # Log progress every 100 images
                if success_count % 100 == 0:
                    logger.info(f"Processed {success_count} images")

            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                error_count += 1

        logger.info(
            f"Extracted embeddings: {success_count} successful, {error_count} failed"
        )

        if success_count == 0 and error_count > 0:
            raise RuntimeError("Failed to extract any embeddings from the dataset")

    def find_similar_images(
        self, target_image_path: str, num_results: int = 5
    ) -> list[tuple[str, float]]:
        """
        Find images similar to the target image using cosine similarity.

        Args:
            target_image_path: Path to the target image.
            num_results: Number of similar images to return (excluding the target image).

        Returns:
            List of tuples containing (image_path, similarity) sorted by similarity,
            with the target image (similarity 1.0) as the first element.

        Raises:
            FileNotFoundError: If the target image does not exist.
            ValueError: If num_results is less than 1.
            RuntimeError: If embedding extraction fails.
        """
        # Input validation
        if not os.path.exists(target_image_path):
            raise FileNotFoundError(f"Target image not found: {target_image_path}")

        if num_results < 1:
            raise ValueError(f"num_results must be at least 1, got {num_results}")

        try:
            # Extract embedding for the target image
            target_embedding = self.extract_features(target_image_path)

            # Extract embeddings for all images if they haven't been extracted yet
            if not self.embeddings:
                logger.info("No embeddings found, extracting now...")
                self.extract_all_embeddings()

            if not self.embeddings:
                raise RuntimeError("Failed to extract embeddings from the dataset")

            # Compare the target embedding with all other embeddings using cosine similarity
            results = []
            for image_path, embedding in self.embeddings.items():
                # Skip comparing the target image with itself
                if os.path.abspath(image_path) == os.path.abspath(target_image_path):
                    continue

                # Calculate cosine similarity
                similarity = cosine_similarity(
                    target_embedding.reshape(1, -1), embedding.reshape(1, -1)
                )[0][0]

                results.append((image_path, float(similarity)))

            # Sort by similarity (highest first)
            results.sort(key=lambda x: x[1], reverse=True)

            # Add the target image at the beginning with similarity 1.0
            final_results = [(target_image_path, 1.0)] + results[:num_results]

            return final_results

        except Exception as e:
            logger.error(f"Error finding similar images to {target_image_path}: {e}")
            raise

    def save_results_to_csv(
        self, results: list[tuple[str, float]], output_path: str
    ) -> None:
        """
        Save the search results to a CSV file.

        Args:
            results: List of tuples containing (image_path, similarity).
            output_path: Path to save the CSV file.

        Raises:
            ValueError: If results is empty.
            IOError: If the CSV file cannot be written.
        """
        # Input validation
        if not results:
            raise ValueError("Cannot save empty results")

        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Create a DataFrame from the results
            df = pd.DataFrame(results, columns=["Filename", "Similarity"])

            # Extract the filename from the full path
            df["Filename"] = df["Filename"].apply(get_filename)

            # Save to CSV
            df.to_csv(output_path, index=False)

            logger.info(f"Results saved to {output_path}")

        except Exception as e:
            logger.error(f"Error saving results to {output_path}: {e}")
            raise IOError(f"Failed to save results to CSV: {e}")
