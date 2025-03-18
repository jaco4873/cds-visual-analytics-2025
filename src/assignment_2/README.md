# Assignment 2: CIFAR-10 Image Classification

This assignment implements classification models for the CIFAR-10 dataset using scikit-learn. The implementation includes two classification approaches:

1. Logistic Regression classifier
2. Neural Network classifier (MLPClassifier)

## Implementation Overview

The implementation follows a modular architecture with the following components:

- **Data Processing**: Loading and preprocessing the CIFAR-10 dataset (conversion to grayscale, normalization)
- **Model Training**: Training both Logistic Regression and Neural Network classifiers
- **Evaluation**: Generating classification reports, confusion matrices, and performance metrics
- **Visualization**: Plotting loss curves (for Neural Network)
- **Configuration-Based**: Uses Pydantic models for flexible configuration

## Project Structure

```
src/assignment_2/
├── README.md                          # This documentation file
├── assignment_description.md          # Original assignment specifications
├── main.py                            # Main script to run classifiers
├── config.py                          # Configuration class for all parameters
├── classifiers/                       # Classification model implementations
│   ├── base_classifier.py             # Abstract base class for classifiers
│   ├── logistic_regression.py         # Logistic Regression implementation
│   └── neural_network.py              # Neural Network implementation
├── utils/                             # Assignment-specific utilities
│   └── cifar_10.py                    # Utilities for CIFAR-10 dataset handling
└── out/                               # Primary results and output files directory
    ├── LogisticRegressionClassifier_confusion_matrix.png
    ├── LogisticRegressionClassifier_info.txt
    ├── LogisticRegressionClassifier_report.txt
    ├── NeuralNetworkClassifier_confusion_matrix.png
    ├── NeuralNetworkClassifier_info.txt
    ├── NeuralNetworkClassifier_loss_curve.png
    └── NeuralNetworkClassifier_report.txt

```

Note: This project also has dependencies on the shared_lib module which provides common utilities and services for image processing, file handling, logging, model evaluation, and visualization.

## Requirements

- Python 3.12
- TensorFlow (for loading the CIFAR-10 dataset)
- scikit-learn
- matplotlib
- numpy
- OpenCV
- pydantic

You can install the required packages with:

```bash
uv sync
```

## How to Run

### Using Configuration-Based Approach

The simplest way to run the classifiers is to use the `main.py` script:

```bash
python -m src.assignment_2.main
```

This will run both classifiers with default settings. The script uses a configuration object to control all aspects of the execution.

### Customizing Configuration

To customize the configuration, override default config by editing the `main.py` file:

```python
# Edit this configuration to change parameters
# Create model-specific configurations
lr_config = LogisticRegressionConfig(
    max_iter=2000,
    solver="saga",
)

nn_config = NeuralNetworkConfig(
    hidden_layer_sizes=(200, 100),  # Two hidden layers
    max_iter=200,
)

# Main configuration
config = CIFAR10Config(
    # Run settings - choose which classifiers to run
    run_models="both",  # Options: "logistic_regression", "neural_network", "both"
    
    # Data processing
    grayscale=True,  # Convert images to grayscale
    normalize=True,  # Normalize pixel values
    
    # Model configurations
    logistic_regression=lr_config,
    neural_network=nn_config,
)
```

## Implementation Details

### Configuration System

The implementation uses a Pydantic-based configuration system that allows:

- Setting all parameters through a single configuration object
- Default values for all parameters
- Type validation and conversion

### Data Preprocessing (default settings)

1. Load the CIFAR-10 dataset using TensorFlow's keras.datasets
2. Convert images to grayscale (optional)
3. Normalize pixel values to [0, 1]
4. Flatten the images to be compatible with scikit-learn classifiers

### Logistic Regression Classifier

The implementation uses scikit-learn's LogisticRegression with the following default parameters:
- Solver: saga (efficient for larger datasets)
- Maximum iterations: 1000
- Tolerance: 0.001
- Regularization parameter (C): 0.1

### Neural Network Classifier

The implementation uses scikit-learn's MLPClassifier with the following default parameters:
- Hidden layer sizes: (200, 100, 50)
- Activation: ReLU
- Solver: adam
- Alpha: 0.0001
- Learning rate: adaptive
- Batch size: 200
- Maximum iterations: 200
- Early stopping: True
- Validation fraction: 0.1 
- n_iter_no_change: 10
- Tolerance: 0.0001

## Output

The classifiers generate the following outputs in the specified output directory:

1. **Classification Report**: Precision, recall, and F1-score for each class
2. **Confusion Matrix**: Visual representation of prediction accuracy
3. **Loss Curve**: Training loss over iterations (Neural Network only)
4. **Model Information**: Model parameters and training details

## Notes on Performance
- The Logistic Regression classifier provides a baseline performance but is limited in capturing complex patterns in image data.
- The Neural Network classifier generally achieves higher accuracy but requires more computational resources.
- Converting to grayscale reduces dimensionality (and training time) but impacts accuracy for color-sensitive classes.

## References

- CIFAR-10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- scikit-learn documentation: https://scikit-learn.org/stable/
- Pydantic documentation: https://docs.pydantic.dev/