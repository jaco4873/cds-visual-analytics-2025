# Visual Analytics Exam Project

This repository contains the exam project for the Visual Analytics course, implementing various computer vision and image analysis techniques.

## Project Overview

The project is organized into multiple assignments, each focusing on different aspects of visual analytics:

- **Assignment 1**: Simple image search algorithm using color histograms
- **Assignment 2**: Image classification benchmarks with Logistic Regression and Neural Networks
- *(Additional assignments will be added as the course progresses)*

## Project Structure

```
.
├── README.md                  # Main documentation file
├── pyproject.toml             # Project configuration and dependencies
├── pytest.ini                 # Configuration for pytest
├── setup.sh                   # Setup script for environment configuration
├── uv.lock                    # Lock file for uv package manager
├── src/                       # Source code directory
│   ├── assignment_1/          # Assignment 1 implementation
│   │   ├── README.md          # Documentation for Assignment 1
│   │   ├── assignment_description.md  # Assignment 1 specifications
│   │   ├── main.py            # Main script for Assignment 1
│   │   └── output/            # Results and output files
│   │       └── results.csv    # Sample output results
│   ├── assignment_2/          # Assignment 2 implementation
│   │   ├── README.md          # Documentation for Assignment 2
│   │   ├── assignment_description.md  # Assignment 2 specifications
│   │   ├── main.py            # Main script for Assignment 2
│   │   ├── config.py          # Configuration for Assignment 2
│   │   ├── classifiers/       # Classification implementations
│   │   │   ├── base_classifier.py     # Abstract base class
│   │   │   ├── logistic_regression.py # Logistic regression implementation
│   │   │   └── neural_network.py      # Neural network implementation
│   │   ├── utils/             # Assignment-specific utilities
│   │   │   └── cifar_10.py    # CIFAR-10 dataset handling
│   │   └── out/               # Output directory
│   ├── data/                  # Data files for assignments (not commited to GH)
│   └── shared_lib/            # Shared utilities and services
│       ├── README.md          # Documentation for shared library
│       ├── services/          # Service classes
│       │   └── image_search.py        # Image search functionality
│       └── utils/             # Utility functions
│           ├── base_config.py         # Base configuration class
│           ├── file_utils.py          # File handling utilities
│           ├── image_utils.py         # Image processing utilities
│           ├── logger.py              # Logging functionality
│           ├── model_evaluation.py    # Model evaluation utilities
│           └── visualization.py       # Visualization utilities
└── tests/                     # Test directory
    └── utils/                 # Tests for utility functions
        ├── test_file_utils.py         # Tests for file utilities
        ├── test_image_utils.py        # Tests for image utilities
        └── test_visualization.py      # Tests for visualization utilities
```

## Requirements

- Python 3.12
- [uv](https://github.com/astral-sh/uv) package manager
- System dependencies:
  - OpenCV
  - Tesseract

## Project Setup

This project uses [uv](https://github.com/astral-sh/uv) for package management and Python 3.12.

### Automatic Setup

```bash
# Run the setup script
source setup.sh
```

Note: The automatic setup only works for macOS / Linux.

The setup script will:
1. Install uv if not already present
2. Check for and install required system dependencies (OpenCV, Tesseract)
3. Create a Python virtual environment
4. Install all project dependencies
5. Activate the virtual environment

### Manual Setup

If you're on Windows or prefer manual setup:

1. Install [uv](https://github.com/astral-sh/uv)
2. Install system dependencies:
   - OpenCV
   - Tesseract
3. Create and activate a virtual environment:
   ```bash
   uv venv --python=3.12
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```
4. Install dependencies:
   ```bash
   uv sync
   ```

## Running the Assignments

### Assignment 1: Simple Image Search

Navigate to the src directory and run:

```bash
cd src
uv run -m assignment_1.main
```

This will:
1. Load the flower dataset from the `data/17flowers` directory
2. Use `image_0001.jpg` as the target image
3. Find the 5 most similar images based on color histogram comparison
4. Save the results to `assignment_1/output/results.csv`
5. Print the results to the console

For more details, see the [Assignment 1 README](src/assignment_1/README.md).

### Assignment 2: CIFAR-10 Image Classification

Run the assignment with:

```bash
# Run both classifiers
uv run -m assignment_2.main

# Run only Logistic Regression
uv run -m assignment_2.main --model logistic_regression

# Run only Neural Network
uv run -m assignment_2.main --model neural_network
```

For more details and configurable options, see the [Assignment 2 README](src/assignment_2/README.md).

## Testing

Tests can be run using pytest:

```bash
pytest
```

The `pytest.ini` file ensures that the project root is in the Python path, allowing imports to work correctly.