"""
Utility functions for path handling and resolution.
"""

import os


def get_project_root() -> str:
    """
    Get the absolute path to the project root directory.

    Returns:
        str: The absolute path to the project root directory.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    return project_root


def resolve_path(relative_path: str) -> str:
    """
    Resolve a path relative to the project root.

    Args:
        relative_path: Path relative to the project root.

    Returns:
        str: The absolute path.
    """
    return os.path.join(get_project_root(), relative_path)


def get_dataset_path(config):
    """Get dataset and target paths from config"""
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    dataset_path = os.path.join(project_root, config.dataset_folder)
    target_path = os.path.join(dataset_path, config.target_image)
    return dataset_path, target_path
