# Visual Analytics Exam Project

This repository contains the exam project for the Visual Analytics course, implementing various computer vision and image analysis techniques.

## Project Overview

The project is organized into multiple assignments, each focusing on different aspects of visual analytics:

- **Assignment 1**: Simple image search algorithm using color histograms
- **Assignment 2**: Image classification benchmarks with Logistic Regression and Neural Networks
- *(Additional assignments will be added as the course progresses)*

## Quickstart

The easiest way to get started is to use the interactive assignment runner:

```bash
# Make the script executable (if needed)
chmod +x run.sh

# Run the interactive assignment runner
./run.sh
```

This script will:
1. Check if the environment is set up, and run setup if needed
2. Check for required datasets and offer to download them automatically if missing
3. Present an interactive menu to select and run any assignment
4. Display the output with clear section markers

**Note:** The interactive assignment runner executes all assignments with their default configurations only. For customized runs with different parameters, please refer to the "Manual Setup and Running" section below or the assignment-specific READMEs.

## Manual Setup and Running

If you prefer to set up and run assignments manually, follow these steps:

### Setup

```bash
# Make the setup script executable (if needed)
chmod +x setup.sh

# Run the setup script
./setup.sh
```

### Running Individual Assignments

After setup, you can run individual assignments:

```bash
# Activate the virtual environment (if not already active)
source .venv/bin/activate

# Run Assignment 1
uv run -m assignment_1.main --method both
uv run -m assignment_1.main --method histogram # Run only histogram-based search
uv run -m assignment_1.main --method embedding # Run only embedding-based search

# Run Assignment 2
cd src && uv run -m assignment_2.main # Run both models
cd src && uv run -m assignment_2.main --model logistic_regression # Run only logistic regression model
cd src && uv run -m assignment_2.main --model neural_network # Run only neural network model
```
**Note:** See assignment-specific README's for more information on running individual assignments and configuration options.

## Assignment Documentation

Each assignment has its own README file with detailed information:

- [Assignment 1 README](src/assignment_1/README.md)
- [Assignment 2 README](src/assignment_2/README.md)

## Project Structure

```.
├── README.md # Main documentation file
├── src/ # Source code directory
│ ├── assignment_1/ # Assignment 1 implementation (image search)
│ │ ├── scripts/ # Implementation scripts for search algorithms
│ │ ├── services/ # Service components for search functionality
│ │ └── output/ # Assignment 1 specific outputs
│ ├── assignment_2/ # Assignment 2 implementation (image classification)
│ │ ├── classifiers/ # Classification model implementations
│ │ ├── utils/ # Assignment-specific utilities
│ │ └── output/ # Output directory for results
│ └── shared_lib/ # Shared utilities used across assignments
└── tests/ # Test directory
```
## Requirements

- Python 3.12
- [uv](https://github.com/astral-sh/uv) package manager
- System dependencies:
  - OpenCV
  - Tesseract

## Testing

Tests can be run using pytest:

```bash
pytest
```

The `pytest.ini` file ensures that the project root is in the Python path, allowing imports to work correctly.

