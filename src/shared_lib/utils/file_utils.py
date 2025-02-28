"""
Utility functions for file operations.
"""

import os


def ensure_directory_exists(directory_path: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        directory_path: Path to the directory.
    """
    os.makedirs(directory_path, exist_ok=True)


def get_file_extension(file_path: str) -> str:
    """
    Get the extension of a file.

    Args:
        file_path: Path to the file.

    Returns:
        The file extension (including the dot).
    """
    return os.path.splitext(file_path)[1].lower()


def get_filename(file_path: str, with_extension: bool = True) -> str:
    """
    Get the filename from a file path.

    Args:
        file_path: Path to the file.
        with_extension: Whether to include the file extension.

    Returns:
        The filename.
    """
    if with_extension:
        return os.path.basename(file_path)
    else:
        return os.path.splitext(os.path.basename(file_path))[0]


def list_files(
    directory: str, extensions: list[str] | None = None, recursive: bool = False
) -> list[str]:
    """
    List files in a directory with optional filtering by extension.

    Args:
        directory: Path to the directory.
        extensions: List of file extensions to include (e.g., ['.jpg', '.png']).
        recursive: Whether to search recursively in subdirectories.

    Returns:
        List of file paths.
    """
    file_list = []

    if recursive:
        for root, _, files in os.walk(directory):
            for file in files:
                if extensions is None or any(
                    file.lower().endswith(ext) for ext in extensions
                ):
                    file_list.append(os.path.join(root, file))
    else:
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path) and (
                extensions is None
                or any(file.lower().endswith(ext) for ext in extensions)
            ):
                file_list.append(file_path)

    return file_list
