# Assignment 1: Image Search with Histograms and Embeddings

## Overview
This project implements two different image search algorithms to find visually similar images:
1. **Histogram-based Search**: Uses color histograms from OpenCV to compare image similarity
2. **Embedding-based Search**: Uses deep learning embeddings from a pre-trained VGG16 model to compare image similarity

## Dataset
The project uses the 17 Category Flower Dataset from the Visual Geometry Group at the University of Oxford. This dataset contains over 1000 images of flowers spanning 17 different species. The full dataset can be accessed from the [official website](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html).

**Note**: The dataset will be automatically downloaded when running the assignment if it's not already present. You don't need to manually download and extract the dataset unless you want to prepare it beforehand.

## Quickstart
If you haven't set up the environment yet, first run the setup script from the project root:

```bash
./setup.sh
```

To run both search algorithms, navigate to the src directory and then execute the main module:

```bash
cd src
uv run -m assignment_1.main
```

You can specify which method to use with the `--method` argument:
```bash
# Run only histogram-based search
uv run -m assignment_1.main --method histogram

# Run only embedding-based search
uv run -m assignment_1.main --method embedding

# Run both methods (default)
uv run -m assignment_1.main --method both
```

The script will:
- Check for the flower dataset and download it automatically if missing
- Load the flower dataset (default: `data/17flowers`)
- Compare images to the target image (default: `image_0001.jpg`)
- Find similar images (default: 5) based on the chosen comparison method(s)
- Save results to output path:
  - Histogram results: `assignment_1/output/histogram_results.csv`
  - Embedding results: `assignment_1/output/embedding_results.csv`

## Configuration
The parameters are defined in `config.py` using a hierarchy of Pydantic BaseSettings classes, which provide early validation of configuration values:

### Base Search Configuration
```python
class BaseSearchConfig(BaseSettings):
    dataset_folder: str = "data/17flowers"
    target_image: str = "image_0001.jpg"
    output_path: str = "assignment_1/output/results.csv"
    num_results: int = 5
```

### Histogram Search Configuration
```python
class HistogramSearchConfig(BaseSearchConfig):
    output_path: str = "assignment_1/output/histogram_results.csv"
    histogram_bins: tuple[int, int, int] = (8, 8, 8)
    color_space: str = "BGR"
```

### Embedding Search Configuration
```python
class EmbeddingSearchConfig(BaseSearchConfig):
    output_path: str = "assignment_1/output/embedding_results.csv"
    input_shape: tuple[int, int, int] = (224, 224, 3)
    pooling: str = "avg"
    include_top: bool = False
```

Due to the simplicity of this assignment, we recommend modifying the default values directly in the `config.py` file rather than using environment variables, although Pydantic supports both approaches.

## Project Structure
```
assignment_1/
├── __init__.py                # Package initialization
├── config.py                  # Configuration settings
├── main.py                    # Main orchestration script
├── services/                  # Core services
│   ├── __init__.py
│   ├── histogram_search_service.py   # Histogram-based search service
│   └── embedding_search_service.py   # Embedding-based search service
└── scripts/                   # Search scripts
    ├── __init__.py
    ├── histogram_search.py    # Histogram search implementation
    └── embedding_search.py    # Embedding search implementation
```

## Implementation Details

### Histogram-based Search
This approach uses color histograms for image similarity comparison:
1. **Color Histogram Extraction**: BGR color space histograms with 8 bins per channel
2. **Comparison**: Chi-Square distance metric (`cv2.HISTCMP_CHISQR`)
3. **Ranking**: Images ranked by histogram distance (lower = more similar)

### Embedding-based Search
This approach uses deep learning embeddings for image similarity comparison:
1. **Feature Extraction**: Pre-trained VGG16 model without top layers to extract image embeddings
2. **Comparison**: Cosine similarity between embeddings
3. **Ranking**: Images ranked by similarity score (higher = more similar)

### Key Components

#### `HistogramSearchService` Class
This service class encapsulates the core functionality for histogram-based search:
- **Initialization**: Sets up the search parameters including image directory, histogram bins, color space, and comparison method
- **Histogram Extraction**: Processes all images in the dataset and stores their histograms
- **Similar Image Search**: Compares the target image histogram with all others and returns the most similar ones
- **Results Export**: Saves the search results to a CSV file

#### `EmbeddingSearchService` Class
This service class encapsulates the core functionality for embedding-based search:
- **Initialization**: Sets up the VGG16 model with the specified parameters
- **Feature Extraction**: Processes images to extract embeddings using the pre-trained model
- **Similar Image Search**: Compares the target image embedding with all others using cosine similarity
- **Results Export**: Saves the search results to a CSV file

### Error Handling
The implementation includes error handling:
- Validates input parameters and paths
- Logs errors and provides informative messages
- Ensures output directories exist before writing files

### Performance Considerations
- Histograms and embeddings are computed once and stored in memory for efficient comparison
- Progress logging for long-running operations like embedding extraction
- The implementation uses OpenCV's optimized functions for histogram extraction and comparison
- TensorFlow's batch prediction for efficient embedding extraction

## Results

### Histogram-based Search
Using the Chi-Square distance metric (lower values indicate more similarity), our histogram-based search found the following most similar images to the target image (image_0001.jpg):

| Rank | Filename       | Distance     |
|------|---------------|--------------|
| 1    | image_0001.jpg | 0.0000       |
| 2    | image_0597.jpg | 4.8699       |
| 3    | image_0594.jpg | 5.0323       |
| 4    | image_0614.jpg | 5.5547       |
| 5    | image_0104.jpg | 5.6719       |
| 6    | image_1126.jpg | 5.6807       |

Manual inspection of these results showed that the histogram approach successfully identified visually similar photos in terms of color composition - most featured yellow flowers in the center with some green in the background, matching the color pattern of the target image.

### Embedding-based Search
Using VGG16 embeddings and cosine similarity (higher values indicate more similarity), our embedding-based search found the following most similar images:

| Rank | Filename       | Similarity   |
|------|---------------|--------------|
| 1    | image_0001.jpg | 1.0000       |
| 2    | image_0037.jpg | 0.8675       |
| 3    | image_0016.jpg | 0.8610       |
| 4    | image_0036.jpg | 0.8394       |
| 5    | image_0017.jpg | 0.8376       |
| 6    | image_0049.jpg | 0.8361       |

What's particularly interesting is that manual inspection of these results revealed all the similar images are yellow daffodils. Unlike the histogram method which simply matched color patterns, the CNN approach recognized the specific flower species in the image. This nicely illustrates how convolutional neural networks can capture semantic content and object categories rather than just low-level visual features.

These results demonstrate how different feature extraction methods can yield different notions of similarity. The histogram-based approach focuses more on color distribution, while the embedding-based approach captures higher-level semantic features, resulting in completely different sets of similar images.