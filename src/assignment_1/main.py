"""
Assignment 1: Image Search with Histograms and Embeddings

This script orchestrates both histogram-based and embedding-based image search
to find similar images in a dataset of flower images.

Author: Jacob Lillelund
Date: 2025-02-28
"""

import os
import click
from shared_lib.logger import logger
from assignment_1.config import histogram_config, embedding_config
from assignment_1.scripts.histogram_search import find_similar_images_with_histograms
from assignment_1.scripts.embedding_search import find_similar_images_with_embeddings


@click.command(help="Image Search with Histograms and Embeddings")
@click.option(
    "--method",
    type=click.Choice(["histogram", "embedding", "both"], case_sensitive=False),
    default="both",
    help="Search method to use (histogram, embedding, or both)",
)
def main(method):
    """
    Execute image search using histograms and/or embeddings.

    This tool finds similar images to a target image using either color histograms,
    deep learning embeddings, or both approaches.
    """
    # Get paths
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    # Common dataset path
    dataset_folder = (
        histogram_config.dataset_folder
    )  # Both configs have the same default
    dataset_path = os.path.join(project_root, dataset_folder)

    try:
        # Run histogram-based search if requested
        if method in ["histogram", "both"]:
            logger.info("=" * 50)
            logger.info("RUNNING HISTOGRAM-BASED IMAGE SEARCH")
            logger.info("=" * 50)

            histogram_target_path = os.path.join(
                dataset_path, histogram_config.target_image
            )

            similar_images_hist = find_similar_images_with_histograms(
                dataset_path=dataset_path,
                target_image_path=histogram_target_path,
                output_path=histogram_config.output_path,
                num_results=histogram_config.num_results,
                histogram_bins=histogram_config.histogram_bins,
                color_space=histogram_config.color_space,
            )

            logger.info("Histogram search results:")
            for i, (image_path, distance) in enumerate(similar_images_hist):
                logger.info(f"{i + 1}. {os.path.basename(image_path)}: {distance:.4f}")

        # Run embedding-based search if requested
        if method in ["embedding", "both"]:
            logger.info("=" * 50)
            logger.info("RUNNING EMBEDDING-BASED IMAGE SEARCH")
            logger.info("=" * 50)

            embedding_target_path = os.path.join(
                dataset_path, embedding_config.target_image
            )

            similar_images_emb = find_similar_images_with_embeddings(
                dataset_path=dataset_path,
                target_image_path=embedding_target_path,
                output_path=embedding_config.output_path,
                num_results=embedding_config.num_results,
                input_shape=embedding_config.input_shape,
                pooling=embedding_config.pooling,
                include_top=embedding_config.include_top,
            )

            logger.info("Embedding search results:")
            for i, (image_path, similarity) in enumerate(similar_images_emb):
                logger.info(
                    f"{i + 1}. {os.path.basename(image_path)}: {similarity:.4f}"
                )

        logger.info("=" * 50)
        logger.info("IMAGE SEARCH COMPLETED SUCCESSFULLY")
        if method == "both":
            logger.info(f"Histogram results saved to: {histogram_config.output_path}")
            logger.info(f"Embedding results saved to: {embedding_config.output_path}")
        elif method == "histogram":
            logger.info(f"Results saved to: {histogram_config.output_path}")
        else:
            logger.info(f"Results saved to: {embedding_config.output_path}")
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"Error during image search: {e}")
        raise


if __name__ == "__main__":
    main()
