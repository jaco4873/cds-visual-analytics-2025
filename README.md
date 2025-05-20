# Visual Analytics Exam Project

This repository contains the exam project for the Visual Analytics course, implementing various computer vision and image analysis techniques.

## Project Overview

The project is organized into multiple assignments, each focusing on different aspects of visual analytics:

- **Assignment 1**: Image search algorithm using color histograms and embedding
- **Assignment 2**: Image classification benchmarks with Logistic Regression and Neural Networks
- **Assignment 3**: Transfer learning with CNNs for Lego brick classification
- **Assignment 4**: Face detection

## Quickstart

Clone the repository and install the required packages:

```bash
# Clone the repository
git clone <repository-url>
cd <repository-directory>

# Install dependencies
uv sync
```

Use the interactive assignment runner:

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

## Setup without using run.sh

If you prefer to set up and run assignments manually, follow these steps:

### Setup

```bash
# Make the setup script executable (if needed)
chmod +x setup.sh

# Run the setup script
./setup.sh
```


After setup, you can run individual assignments:

```bash
# Activate the virtual environment (if not already active)
source .venv/bin/activate

# Navigate to src directory
cd src

# Run Assignment 1
uv run python -m assignment_1.main --method both
uv run python -m assignment_1.main --method histogram # Run only histogram-based search
uv run python -m assignment_1.main --method embedding # Run only embedding-based search

# Run Assignment 2
uv run python -m assignment_2.main # Run both models
uv run python -m assignment_2.main --model logistic_regression # Run only logistic regression model
uv run python -m assignment_2.main --model neural_network # Run only neural network model

# Run Assignment 3
uv run python -m assignment_3.main # Run both CNN and VGG16 models
uv run python -m assignment_3.main --cnn_only # Run only the direct CNN model
uv run python -m assignment_3.main --vgg16_only # Run only the VGG16 transfer learning model

# Run Assignment 4
uv run python -m assignment_4.main # Run face detection
```
**Note:** See assignment-specific README's for more information on running individual assignments and configuration options.

## Assignment Documentation

Each assignment has its own README file with detailed information:

- [Assignment 1 README](src/assignment_1/README.md)
- [Assignment 2 README](src/assignment_2/README.md)
- [Assignment 3 README](src/assignment_3/README.md)
- [Assignment 4 README](src/assignment_4/README.md)

## Project Structure

```
├── README.md                  # Main documentation file
├── assignment_1/              # Assignment 1 implementation (image search)
│   ├── README.md              # Assignment-specific documentation
│   ├── assignment_description.md # Original assignment details
│   ├── check_and_download_data.sh # Data download script
│   ├── config.py              # Configuration settings
│   ├── main.py                # Entry point for assignment 1
│   ├── output/                # Assignment 1 specific outputs
│   ├── scripts/               # Implementation scripts for search algorithms
│   ├── services/              # Service components for search functionality
│   └── utils/                 # Image utility functions
├── assignment_2/              # Assignment 2 implementation (image classification)
│   ├── README.md              # Assignment-specific documentation
│   ├── assignment_description.md # Original assignment details
│   ├── config.py              # Configuration settings
│   ├── main.py                # Entry point for assignment 2
│   ├── models/                # Classification model implementations
│   ├── output/                # Output directory for results
│   └── utils/                 # Assignment-specific utilities
├── assignment_3/              # Assignment 3 implementation (transfer learning)
│   ├── README.md              # Assignment-specific documentation
│   ├── assignment_description.md # Original assignment details
│   ├── config.py              # Configuration settings
│   ├── data/                  # Data loading functionality
│   ├── main.py                # Entry point for assignment 3
│   ├── models/                # CNN and VGG16 model implementations
│   ├── output/                # Output directory for model results
│   └── utils/                 # Utility functions for model comparison
├── assignment_4/              # Assignment 4 implementation (face detection)
│   ├── README.md              # Assignment-specific documentation
│   ├── assignment_description.md # Original assignment details
│   ├── config.py              # Configuration settings
│   ├── data/                  # Data service components
│   ├── main.py                # Entry point for assignment 4
│   ├── models/                # Face detection model implementation
│   ├── output/                # Output directory
│   │   ├── plots/             # Visualization plots
│   │   └── results/           # Analysis results
│   └── visualization/         # Visualization components
└── shared_lib/                # Shared utilities used across assignments
```