"""
Pytest tests for file utility functions.
"""

import os
import pytest
from src.shared_lib.utils.file_utils import (
    ensure_directory_exists,
    get_file_extension,
    get_filename,
    list_files,
)


# Fixtures
@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing."""
    temp_dir = "temp_test_dir"

    # Create the directory
    os.makedirs(temp_dir, exist_ok=True)

    # Return directory path and cleanup function
    yield temp_dir

    # Cleanup after test
    if os.path.exists(temp_dir):
        os.rmdir(temp_dir)


@pytest.fixture
def temp_files():
    """Create temporary files with different extensions for testing."""
    temp_dir = "temp_test_files"
    os.makedirs(temp_dir, exist_ok=True)

    # Create files with different extensions
    file_paths = []
    for ext in [".txt", ".jpg", ".png", ".pdf"]:
        file_path = os.path.join(temp_dir, f"test_file{ext}")
        with open(file_path, "w") as f:
            f.write(f"Test content for {ext} file")
        file_paths.append(file_path)

    # Create a subdirectory with files
    subdir = os.path.join(temp_dir, "subdir")
    os.makedirs(subdir, exist_ok=True)

    # Add files to subdirectory
    for ext in [".jpg", ".txt"]:
        file_path = os.path.join(subdir, f"subdir_file{ext}")
        with open(file_path, "w") as f:
            f.write(f"Subdir test content for {ext} file")
        file_paths.append(file_path)

    # Return directory path, file paths, and cleanup function
    yield temp_dir, file_paths

    # Cleanup after test
    for path in file_paths:
        if os.path.exists(path):
            os.remove(path)

    if os.path.exists(subdir):
        os.rmdir(subdir)

    if os.path.exists(temp_dir):
        os.rmdir(temp_dir)


# Tests for ensure_directory_exists
def test_ensure_directory_exists_new():
    """Test creating a new directory."""
    test_dir = "test_new_directory"

    # Make sure the directory doesn't exist
    if os.path.exists(test_dir):
        os.rmdir(test_dir)

    try:
        # Test function
        ensure_directory_exists(test_dir)

        # Check that directory was created
        assert os.path.exists(test_dir)
        assert os.path.isdir(test_dir)
    finally:
        # Clean up
        if os.path.exists(test_dir):
            os.rmdir(test_dir)


def test_ensure_directory_exists_existing(temp_directory):
    """Test with an existing directory."""
    # Directory already exists from fixture
    ensure_directory_exists(temp_directory)

    # Check that directory still exists
    assert os.path.exists(temp_directory)
    assert os.path.isdir(temp_directory)


def test_ensure_directory_exists_nested():
    """Test creating nested directories."""
    test_nested_dir = os.path.join("test_parent", "test_child", "test_grandchild")

    try:
        # Test function
        ensure_directory_exists(test_nested_dir)

        # Check that all directories were created
        assert os.path.exists(test_nested_dir)
        assert os.path.isdir(test_nested_dir)
    finally:
        # Clean up - remove directories from innermost to outermost
        if os.path.exists(os.path.join("test_parent", "test_child", "test_grandchild")):
            os.rmdir(os.path.join("test_parent", "test_child", "test_grandchild"))

        if os.path.exists(os.path.join("test_parent", "test_child")):
            os.rmdir(os.path.join("test_parent", "test_child"))

        if os.path.exists("test_parent"):
            os.rmdir("test_parent")


# Tests for get_file_extension
def test_get_file_extension_normal():
    """Test getting extension from normal file paths."""
    test_cases = [
        ("file.txt", ".txt"),
        ("path/to/image.jpg", ".jpg"),
        ("document.PDF", ".pdf"),  # Should be lowercase
        ("script.py", ".py"),
        ("/absolute/path/to/file.docx", ".docx"),
    ]

    for file_path, expected in test_cases:
        assert get_file_extension(file_path) == expected


def test_get_file_extension_special_cases():
    """Test getting extension from special file paths."""
    test_cases = [
        ("file", ""),  # No extension
        (".hidden", ""),  # Hidden file without extension
        ("path/to/.gitignore", ""),  # Hidden file without extension
        ("file.with.multiple.dots.txt", ".txt"),  # Multiple dots
        ("path.to/file", ""),  # Dot in directory name
    ]

    for file_path, expected in test_cases:
        assert get_file_extension(file_path) == expected


# Tests for get_filename
def test_get_filename_with_extension():
    """Test getting filename with extension."""
    test_cases = [
        ("file.txt", "file.txt"),
        ("path/to/image.jpg", "image.jpg"),
        ("/absolute/path/to/document.pdf", "document.pdf"),
        ("./relative/script.py", "script.py"),
        ("file_without_extension", "file_without_extension"),
    ]

    for file_path, expected in test_cases:
        assert get_filename(file_path, with_extension=True) == expected


def test_get_filename_without_extension():
    """Test getting filename without extension."""
    test_cases = [
        ("file.txt", "file"),
        ("path/to/image.jpg", "image"),
        ("/absolute/path/to/document.pdf", "document"),
        ("./relative/script.py", "script"),
        ("file_without_extension", "file_without_extension"),
        ("file.with.multiple.dots.txt", "file.with.multiple.dots"),
    ]

    for file_path, expected in test_cases:
        assert get_filename(file_path, with_extension=False) == expected


# Tests for list_files
def test_list_files_all(temp_files):
    """Test listing all files in a directory."""
    temp_dir, all_files = temp_files

    # Get all files in the main directory (not recursive)
    files = list_files(temp_dir)

    # Should find 4 files in the main directory
    assert len(files) == 4

    # All returned paths should be in the main directory
    for file in files:
        assert os.path.dirname(file) == temp_dir


def test_list_files_with_extensions(temp_files):
    """Test listing files with specific extensions."""
    temp_dir, _ = temp_files

    # Get only image files
    image_files = list_files(temp_dir, extensions=[".jpg", ".png"])

    # Should find 2 image files in the main directory
    assert len(image_files) == 2

    # All returned files should have the specified extensions
    for file in image_files:
        ext = os.path.splitext(file)[1].lower()
        assert ext in [".jpg", ".png"]


def test_list_files_recursive(temp_files):
    """Test listing files recursively."""
    temp_dir, all_files = temp_files

    # Get all files recursively
    files = list_files(temp_dir, recursive=True)

    # Should find all 6 files (4 in main dir + 2 in subdir)
    assert len(files) == 6

    # Check that files from subdirectory are included
    subdir_files = [f for f in files if "subdir" in f]
    assert len(subdir_files) == 2


def test_list_files_recursive_with_extensions(temp_files):
    """Test listing files recursively with specific extensions."""
    temp_dir, _ = temp_files

    # Get only .txt files recursively
    txt_files = list_files(temp_dir, extensions=[".txt"], recursive=True)

    # Should find 2 .txt files (1 in main dir + 1 in subdir)
    assert len(txt_files) == 2

    # All returned files should have .txt extension
    for file in txt_files:
        assert file.lower().endswith(".txt")


def test_list_files_empty_directory():
    """Test listing files in an empty directory."""
    # Create a temporary empty directory
    empty_dir = "temp_empty_dir"
    os.makedirs(empty_dir, exist_ok=True)

    try:
        files = list_files(empty_dir)
        assert len(files) == 0
    finally:
        # Clean up
        if os.path.exists(empty_dir):
            os.rmdir(empty_dir)


def test_list_files_nonexistent_directory():
    """Test listing files in a non-existent directory."""
    with pytest.raises(FileNotFoundError):
        list_files("non_existent_directory")
