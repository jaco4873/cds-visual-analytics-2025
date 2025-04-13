# Assignment 1: Simple Image Search Algorithm

## Overview
This project implements an image search algorithm that finds visually similar images based on color histograms using OpenCV.

## Dataset
The project uses the 17 Category Flower Dataset from the Visual Geometry Group at the University of Oxford. This dataset contains over 1000 images of flowers spanning 17 different species. The full dataset can be accessed from the [official website](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html).

**Note**: The dataset will be automatically downloaded when running the assignment if it's not already present. You don't need to manually download and extract the dataset unless you want to prepare it beforehand.

## Quickstart
If you haven't set up the environment yet, first run the setup script from the project root:

```bash
./setup.sh
```

To run the orchestration script main.py, navigate to the src directory and then execute the module:

```python
cd src
uv run -m assignment_1.main
```

The script will:
- Check for the flower dataset and download it automatically if missing
- Load the flower dataset (default: `data/17flowers`)
- Compare images to the target image (default: `image_0001.jpg`)
- Find similar images (default: 5) based on color histogram comparison
- Save results to output path (default: `assignment_1/output/results.csv`)

The parameters are defined in `config.py` using Pydantic's BaseSettings, which provides early validation of configuration values:

```python
# Configuration parameters in config.py
dataset_folder: str = "data/17flowers"
target_image: str = "image_0001.jpg"
output_path: str = "assignment_1/output/results.csv"
num_results: int = 5
```

Due to the simplicity of this assignment, we recommend modifying the default values directly in the `config.py` file rather than using environment variables, although Pydantic supports both approaches.

## Project Structure
- `main.py`: Orchestration script for the image search workflow
- `image_search_service.py`: Core `ImageSearchService` functionality
- `config.py`: Configuration settings using Pydantic
- `download_data.py`: Script to automatically download and extract the dataset when needed
- `shared_lib/utils/`: Utility functions for file and image operations

## Implementation Details
The solution uses color histograms for image similarity comparison:
1. **Color Histogram Extraction**: BGR color space histograms with 8 bins per channel
2. **Comparison**: Chi-Square distance metric (`cv2.HISTCMP_CHISQR`)
3. **Ranking**: Images ranked by histogram distance (lower = more similar)

### Key Components

#### `ImageSearchService` Class

This service class encapsulates the core functionality for image search:
- **Initialization**: Sets up the search parameters including image directory, histogram bins, color space, and comparison method
- **Histogram Extraction**: Processes all images in the dataset and stores their histograms
- **Similar Image Search**: Compares the target image histogram with all others and returns the most similar ones
- **Results Export**: Saves the search results to a CSV file

### Error Handling
The implementation includes error handling:
- Validates input parameters and paths
- Logs errors and provides informative messages
- Ensures output directories exist before writing files

### Performance Considerations
- Histograms are computed once and stored in memory for efficient comparison
- The implementation uses OpenCV's optimized functions for histogram extraction and comparison

## Results
Results show the target image (image_0001.jpg) has the closest matches with image_0597.jpg (distance: 4.87), image_0594.jpg (distance: 5.03), and image_0614.jpg (distance: 5.55). 
The Chi-Square distance metric indicates lower values for more similar images.