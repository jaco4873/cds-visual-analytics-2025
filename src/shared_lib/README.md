# Shared Library

A collection of reusable components for image processing and analysis used throughout the Visual Analytics Exam Project.

## Purpose

This library centralizes common functionality across all assignments to:
- Eliminate code duplication
- Ensure consistent implementation of core features
- Provide a unified interface for common operations
- Simplify maintenance and updates

## Components

### Services

Service classes that encapsulate complex functionality:
- **Image Search**: Histogram-based similarity search
- **Image Processing**: Common transformations and feature extraction
- **Data Management**: Dataset handling and result storage

### Utilities

Helper functions for routine operations:
- **File Operations**: Reading/writing files, directory management
- **Image Handling**: Loading, saving, and basic transformations
- **Visualization**: Result plotting and image display
- **Metrics**: Evaluation and comparison metrics
- **Logging**: Standardized logging and error handling

## Usage

Import specific components as needed in your assignment scripts:

```python
# Import services
from shared_lib.services.image_search import ImageSearchService
from shared_lib.services.image_processor import ImageProcessor

# Import utilities
from shared_lib.utils.file_utils import ensure_dir_exists
from shared_lib.utils.visualization import plot_results
```