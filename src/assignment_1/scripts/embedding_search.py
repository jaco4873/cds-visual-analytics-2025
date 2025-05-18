"""
Assignment 1: Image Search with Embeddings

This script implements an image search algorithm using VGG16 embeddings
to find similar images in a dataset of flower images.

Author: Jacob Lillelund
Date: 2025-02-28
"""

import os
from assignment_1.services.embedding_search_service import EmbeddingSearchService
from assignment_1.config import embedding_config
from shared_lib.logger import logger
from assignment_1.utils.path_utils import get_dataset_path
from assignment_1.utils.result_utils import save_results_to_csv


def find_similar_images_with_embeddings(
    dataset_path: str,
    target_image_path: str,
    output_path: str,
    num_results: int = 5,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    pooling: str = "avg",
    include_top: bool = False,
) -> list[tuple[str, float]]:
    """
    Find images similar to a target image based on VGG16 embeddings.

    Args:
        dataset_path (str): Path to the directory containing the image dataset.
        target_image_path (str): Path to the target image.
        output_path (str): Path to save the results CSV file.
        num_results (int): Number of similar images to find.
        input_shape (tuple): Input shape for the CNN model (height, width, channels).
        pooling (str): Pooling method for the CNN model.
        include_top (bool): Whether to include the fully connected layers.

    Returns:
        list: List of tuples containing (image_path, similarity) sorted by similarity.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Initialize the embedding search service
    logger.info(f"Initializing embedding search service with dataset: {dataset_path}")
    search_service = EmbeddingSearchService(
        image_directory=dataset_path,
        input_shape=input_shape,
        pooling=pooling,
        include_top=include_top,
    )

    # Extract embeddings for all images in the dataset
    logger.info("Extracting embeddings for all images...")
    search_service.extract_all_embeddings()

    # Find similar images
    logger.info(f"Finding {num_results} images similar to {target_image_path}...")
    similar_images = search_service.find_similar_images(
        target_image_path=target_image_path,
        num_results=num_results,
    )

    # Save results to CSV
    logger.info(f"Saving results to {output_path}...")
    save_results_to_csv(
        results=similar_images, output_path=output_path, metric_name="Similarity"
    )

    logger.info("Embedding image search completed successfully!")
    return similar_images


def main() -> None:
    """
    Main function to execute the embedding-based image search workflow.

    This function defines the parameters for the image search and calls
    the find_similar_images_with_embeddings function.
    """
    dataset_path, target_image_path = get_dataset_path(embedding_config)

    try:
        # Find similar images using embeddings
        logger.info(f"Dataset path: {dataset_path}")
        logger.info(f"Target image path: {target_image_path}")
        logger.info(f"Output path: {embedding_config.output_path}")
        logger.info(f"Number of results: {embedding_config.num_results}")
        logger.info("Starting embedding-based image search. This may take a while...")

        similar_images = find_similar_images_with_embeddings(
            dataset_path=dataset_path,
            target_image_path=target_image_path,
            output_path=embedding_config.output_path,
            num_results=embedding_config.num_results,
            input_shape=embedding_config.input_shape,
            pooling=embedding_config.pooling,
            include_top=embedding_config.include_top,
        )

        logger.info("Embedding-based image search completed successfully!")

        # Log results for verification
        logger.info("Most similar images (based on embeddings):")
        for i, (image_path, similarity) in enumerate(similar_images):
            logger.info(f"{i + 1}. {os.path.basename(image_path)}: {similarity:.4f}")

    except Exception as e:
        logger.error(f"Error during embedding-based image search: {e}")
        raise


if __name__ == "__main__":
    main()
