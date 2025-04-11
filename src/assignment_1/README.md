# Assignment 1: Simple Image Search Algorithm

## Overview

This project implements a simple image search algorithm that finds visually similar images based on color histograms. The implementation follows the requirements specified in the assignment description, using OpenCV to extract and compare color histograms from a dataset of flower images.

## Quickstart
To run the orchestration script main.py, navigate to the src directory and then execute the module:

```python
cd src
uv run -m assignment_1.main
```

The script will:
1. Load the flower dataset from the configured dataset folder (default: `data/17flowers`)
2. Use the configured target image (default: `image_0001.jpg`)
3. Find the configured number of similar images (default: 5) based on color histogram comparison
4. Save the results to the configured output path (default: `assignment_1/output/results.csv`)
5. Print the results to the console

The parameters can be modified by changing the configuration variables at the top of the `main.py` file:
```python
# Configuration parameters
DATASET_FOLDER = "data/17flowers"
TARGET_IMAGE = "image_0001.jpg"
OUTPUT_PATH = "assignment_1/output/results.csv"
NUM_RESULTS = 5
```

## Project Structure

- `main.py`: The main script that orchestrates the image search workflow
- `shared_lib/services/image_search.py`: Contains the `ImageSearchService` class that handles the core functionality
- `shared_lib/utils/`: Contains utility functions for file operations, image processing, and logging

## Implementation Details

### Image Search Approach

The solution uses color histograms as a simple feature for comparing image similarity. The approach works as follows:

1. **Color Histogram Extraction**: For each image, a 3D color histogram is computed in the BGR color space with 8 bins per channel.
2. **Histogram Comparison**: The Chi-Square distance metric (`cv2.HISTCMP_CHISQR`) is used to measure the similarity between histograms.
3. **Ranking**: Images are ranked based on their histogram distance to the target image, with lower distances indicating higher similarity.

### Key Components

#### `ImageSearchService` Class

This service class encapsulates the core functionality for image search:

- **Initialization**: Sets up the search parameters including image directory, histogram bins, color space, and comparison method
- **Histogram Extraction**: Processes all images in the dataset and stores their histograms
- **Similar Image Search**: Compares the target image histogram with all others and returns the most similar ones
- **Results Export**: Saves the search results to a CSV file

#### Main Workflow

The `main.py` script implements the following workflow:

1. **Setup**: Defines paths for the dataset, target image, and output file
2. **Service Initialization**: Creates an instance of `ImageSearchService` with appropriate parameters
3. **Histogram Extraction**: Processes all images in the dataset to extract their color histograms
4. **Similar Image Search**: Finds the 5 most similar images to the target image
5. **Results Export**: Saves the results to a CSV file and prints them to the console

## Implementation Highlights

### Error Handling

The implementation includes robust error handling:
- Validates input parameters and paths
- Logs errors and provides informative messages
- Ensures output directories exist before writing files

### Modularity

The code is organized in a modular way:
- The `ImageSearchService` class encapsulates the core functionality
- Utility functions handle common operations like file I/O and image processing
- The main script orchestrates the workflow without duplicating functionality

### Performance Considerations

- Histograms are computed once and stored in memory for efficient comparison
- The implementation uses OpenCV's optimized functions for histogram extraction and comparison