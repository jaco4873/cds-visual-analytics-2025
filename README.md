# Visual Analytics Exam Project

This repository contains the exam project for the Visual Analytics course, implementing various computer vision and image analysis techniques.

## Project Overview

The project is organized into multiple assignments, each focusing on different aspects of visual analytics:

- **Assignment 1**: Simple image search algorithm using color histograms
- **Assignment 2**: Image classification benchmarks with Logistic Regression and Neural Networks
- **Assignment 3**: Transfer learning with pretrained CNNs for Lego brick classification
- **Assignment 4**: Face detection
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

# Run Assignment 3
cd src && uv run -m assignment_3.main # Run both CNN and VGG16 models
cd src && uv run -m assignment_3.main --cnn_only # Run only the direct CNN model
cd src && uv run -m assignment_3.main --vgg16_only # Run only the VGG16 transfer learning model

# Run Assignment 4
cd src && uv run -m assignment_4.main # Run face detection
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
├── src/                       # Source code directory
│   ├── assignment_1/          # Assignment 1 implementation (image search)
│   │   ├── scripts/           # Implementation scripts for search algorithms
│   │   ├── services/          # Service components for search functionality
│   │   └── output/            # Assignment 1 specific outputs
│   ├── assignment_2/          # Assignment 2 implementation (image classification)
│   │   ├── classifiers/       # Classification model implementations
│   │   ├── utils/             # Assignment-specific utilities
│   │   └── output/            # Output directory for results
│   ├── assignment_3/          # Assignment 3 implementation (transfer learning)
│   │   ├── services/          # Service components for CNN and VGG16 models
│   │   └── output/            # Output directory for model results
│   ├── assignment_4/          # Assignment 4 implementation (face detection)
│   │   ├── services/          # Service components for data and face detection
│   │   ├── utils/             # Visualization utilities
│   │   └── output/            # Output directory for results and plots
│   └── shared_lib/            # Shared utilities used across assignments
```