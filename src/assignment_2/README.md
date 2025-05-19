# Assignment 2: CIFAR-10 Image Classification

This assignment implements classification models for the CIFAR-10 dataset using scikit-learn. The implementation includes two classification approaches:

1. Logistic Regression classifier
2. Neural Network classifier (MLPClassifier)

## Quickstart

The simplest way to run the assignment is using the provided run.sh script:
```bash
./run.sh
```
Then select option 2 from the menu.

**Note:** Using the run.sh script will execute the assignment with default configurations only. For customized runs with different parameters, see the Advanced Execution Options section below.

### Manual Execution
To run the assignment directly, navigate to the src directory and execute:

```bash
cd src
uv run python -m assignment_2.main
```

#### Customizing Configuration

There are two ways to customize the configuration:

1. **Edit config.py:** Modify the default configuration values directly in the `config.py` file.
2. **Use command line options:** Override specific settings using command line flags when running the script.

You can customize the run using command line options which will override the corresponding settings in `config.py`.
The main.py script accepts the following commands:

```
--model  Which model to run [logistic_regression, neural_network, both]
         If not specified, uses the value from config.py (default: "both")
```

Example usage:
```bash
cd src
uv run python -m assignment_2.main --model neural_network
```

When the `--model` flag is provided, it overrides the `run_models` setting in the configuration. If not provided, the script uses the value from `config.py`.

## Project Structure

```
src/assignment_2/
├── README.md                          # This documentation file
├── main.py                            # Main script to run classifiers
├── config.py                          # Configuration class for all parameters
├── models/                            # Classification model implementations
│   ├── base_classifier.py             # Abstract base class for classifiers
│   ├── logistic_regression.py         
│   └── neural_network.py              
├── utils/                             # Assignment-specific utilities
│   ├── cifar_10.py                  
│   ├── image_utils.py                 
│   └── model_evaluation.py                
└── output/                            # Primary results and output files directory
        ...

```

Note: This project also has dependencies on the shared_lib module which provides common utilities and services for image processing, file handling, logging, model evaluation, and visualization.



## Implementation Overview

The implementation follows a modular architecture with the following components:

- **Data Processing**: Loading and preprocessing the CIFAR-10 dataset (conversion to grayscale, normalization)
- **Model Training**: Training both Logistic Regression and Neural Network classifiers
- **Evaluation**: Generating classification reports, confusion matrices, and performance metrics
- **Visualization**: Plotting loss curves (for Neural Network)
- **Configuration-Based**: Uses Pydantic models for flexible configuration

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

Note that in scikit-learn's LogisticRegression:
- `max_iter` refers to the maximum number of solver iterations for convergence (not epochs)
- The solver will stop earlier if the model converges (when the improvement in loss is less than `tol`)
- The actual number of iterations is typically lower than the maximum when convergence is reached

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

Note that in scikit-learn's MLPClassifier:
- `max_iter` directly corresponds to the maximum number of epochs (complete passes through the training data)
- With `early_stopping` enabled, training may stop before reaching the maximum number of epochs
- Training stops after `n_iter_no_change` consecutive epochs without improvement on the validation set
- The actual number of epochs conducted is typically lower than the maximum (as shown in the Results Analysis section, the model stopped after 50 epochs due to early stopping)

### Data Splitting

It's worth noting how the data is split differently between the models:

1. **Initial Split**: The CIFAR-10 dataset comes pre-divided into 50,000 training images and 10,000 test images.

2. **Neural Network Classifier** (with early stopping enabled):
   - Creates a three-way split of the data
   - Training set: 45,000 images (90% of original training data)
   - Validation set: 5,000 images (10% of original training data, controlled by validation_fraction parameter)
   - Test set: 10,000 images (original test set)
   - The validation set is created internally by scikit-learn's MLPClassifier and used to monitor performance for early stopping

3. **Logistic Regression Classifier**:
   - Uses a simple two-way split
   - Training set: All 50,000 training images
   - Test set: 10,000 images (original test set)
   - No validation set is needed since there's no early stopping mechanism

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

## Results Analysis

The experiments on the CIFAR-10 dataset yielded interesting findings:

### Logistic Regression Classifier
- Achieved an overall accuracy of 29.59% on the test set
- Best performance on truck (40.28% F1-score) and automobile (36.84% F1-score) classes
- Struggled most with cat (18.64% F1-score) and deer (20.83% F1-score) classes
- Converged after 121 iterations with the saga solver

### Neural Network Classifier
- Achieved a significantly higher overall accuracy of 43.85%
- Best performance on truck (51.44% F1-score), ship (51.24% F1-score), and automobile (50.58% F1-score) classes
- Still struggled with cats (24.88% F1-score), suggesting this class is inherently challenging
- Early stopping triggered after 50 iterations

Looking at both models, it's clear the neural network does a much better job with these complex images, even when converted to grayscale. It's interesting that cats were consistently the hardest class for both models to identify. I noticed vehicles (cars, trucks, ships) were the easiest to classify across both models.

## Requirements

- Python 3.12
- TensorFlow (for loading the CIFAR-10 dataset)
- scikit-learn
- matplotlib
- numpy
- OpenCV
- pydantic
- Click


## References

- CIFAR-10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- scikit-learn documentation: https://scikit-learn.org/stable/
- Pydantic documentation: https://docs.pydantic.dev/