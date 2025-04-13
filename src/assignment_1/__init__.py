"""
Assignment 1: Image Search with Histograms and Embeddings

This package contains the implementation of two image search algorithms:
1. Histogram-based search using OpenCV
2. Embedding-based search using VGG16
"""

from assignment_1.config import histogram_config, embedding_config
from assignment_1.scripts import (
    find_similar_images_with_histograms,
    find_similar_images_with_embeddings,
)
from assignment_1.services import HistogramSearchService, EmbeddingSearchService

__all__ = [
    "histogram_config",
    "embedding_config",
    "find_similar_images_with_histograms",
    "find_similar_images_with_embeddings",
    "HistogramSearchService",
    "EmbeddingSearchService",
]
