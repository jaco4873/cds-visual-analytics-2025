---
title: "Visual Analytics Portfolio"
author: "Jacob Lillelund"
date: "20 May 2025"
titlepage: true
titlepage-color: "003366"
titlepage-text-color: "FFFFFF"
titlepage-rule-color: "FFFFFF"
titlepage-rule-height: 2
toc-own-page: true
monofont: "Menlo"
fontsize: 12pt
---

# Visual Analytics Exam Project

This repository contains the exam project for the Visual Analytics course, implementing various computer vision and image analysis techniques.

## Project Overview

The project is organized into multiple assignments, each focusing on different aspects of visual analytics:

- **Assignment 1**: Image search algorithm using color histograms and embedding search
- **Assignment 2**: Image classification with Logistic Regression and Neural Network
- **Assignment 3**: Transfer learning with CNNs for Lego brick classification
- **Assignment 4**: Face detection with MTCNN

## Data Preparation

Assignments includes specific guides on how to obtain the necessary datasets where relevant. All data should be placed in the top-level `/data` directory (outside the `/src` directory) and have this overall structure:

```
data/
├── 17flowers/    # Flower dataset used in Assignment 1
├── lego/         # Lego brick dataset used in Assignment 3
└── newspapers/   # Newspaper dataset used in Assignment 4
```

## Quickstart

Use the interactive assignment runner:

```bash
# Make the script executable (if needed)
chmod +x run.sh

# Run the interactive assignment runner
./run.sh
```

This script will:
1. Check if the environment is setup, and if not, prompts queries whether the user would like to run `setup.py`
2. Present an interactive menu to select and run any assignment (after environment setup completes)
3. Display and point to results after an assignment is run.

**Note:** The interactive assignment runner executes all assignments with their default configurations only. For customized runs with different parameters, please refer to the "Manual Setup and Running" section below or the assignment-specific READMEs.

### Setup without using run.sh

If you prefer to set up and run assignments manually, follow these steps:

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

## Assignment Specific Documentation

Each assignment has its own README file with detailed information:

- [Assignment 1 README](src/assignment_1/README.md)
- [Assignment 2 README](src/assignment_2/README.md)
- [Assignment 3 README](src/assignment_3/README.md)
- [Assignment 4 README](src/assignment_4/README.md)