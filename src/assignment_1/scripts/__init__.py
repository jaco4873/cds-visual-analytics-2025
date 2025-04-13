"""
Scripts package for image search functionality.
"""

from assignment_1.scripts.histogram_search import find_similar_images_with_histograms
from assignment_1.scripts.embedding_search import find_similar_images_with_embeddings

__all__ = ["find_similar_images_with_histograms", "find_similar_images_with_embeddings"]
