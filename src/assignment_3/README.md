# Assignment 3: Transfer Learning with Pretrained CNNs

## Overview
This assignment implements two approaches for classifying Lego brick images:
1. A custom CNN trained directly on the image data
2. A transfer learning approach using VGG16 as a feature extractor

The goal is to compare these two approaches and determine whether transfer learning improves performance for this specific image classification task.

## Quickstart

The simplest way to run the assignment is using the provided run.sh script:

```bash
./run.sh
```
Then select option 3 from the menu.

### Manual Execution
You can also run the code directly:
```bash
# Navigate to src from the project root
cd src

# Run as module
uv run python -m assignment_3.main 

# With custom arguments
uv run python -m assignment_3.main --data-dir /path/to/lego/data --output-dir /path/to/output
```

#### Command Line Arguments
- `--data-dir`: Path to the Lego data directory (must exist and be a directory)
- `--output-dir`: Path to save the output (default: ./src/assignment_3/output)
- `--cnn-only`: Train only the CNN model
- `--vgg16-only`: Train only the VGG16 model

## Project Structure
```
src/assignment_3/
├── config.py                    # Configuration settings
├── main.py                      # Main entry point
├── assignment_description.md    # Assignment description
├── README.md                    # Project documentation
├── session9_inclass_rdkm.ipynb  # In-class notebook reference
├── data/                        # Data modules
│   └── data_loader.py           # Data loader
├── models/                      # Service modules
│   ├── base_classifier_model.py # Base model class
│   ├── cnn_model.py             # Direct CNN classifier
│   └── vgg16_transfer_learning_model.py  # VGG16-based classifier
├── utils/                       # Utility functions
│   └── model_comparison.py      # Model comparison utilities
└── output/                      # Output directory
    ├── cnn/                     # Output for CNN model
    │   ├── classification_report.txt
    │   ├── learning_curves.png
    │   └── training_history.json
    ├── vgg16/                   # Output for VGG16 model
    │   ├── classification_report.txt
    │   ├── learning_curves.png
    │   └── training_history.json
    ├── model_comparison.png     # Visual comparison of models
    └── model_comparison.txt     # Textual comparison of models
```

## Dataset
- Source and format: The dataset consists of images of various Lego bricks arranged in folders named after the brick type. 
- The data should be placed in the `data/lego` directory. 
- We use the data from the "Cropped Images" folder, which contains images with backgrounds removed.

## Implementation Details

### Data Processing
- Images are resized to 224×224 pixels with 3 color channels
- Batches of 32 images are processed at a time
- Data is split with 20% for validation during training
- Image normalization scales pixel values to [0-1]
- Data augmentation is applied to improve model robustness:
  - Horizontal flipping

### CNN Model Architecture
The custom CNN is configured with the following architecture:
- Input shape: (224, 224, 3)
- Convolutional layers with increasing filter complexity: [32, 64, 128]
- 3×3 kernel sizes with 2×2 max pooling
- Two dense layers [512, 256] with ReLU activation
- Dropout (0.5) for regularization
- Softmax output layer for classification
- Adam optimizer with learning rate 0.001
- Trained for 15 epochs

### VGG16 Transfer Learning Model
- We're using VGG16 (pretrained on ImageNet) as our foundation
- We've removed the top classification layers (include_top=False)
- **We're fine-tuning the last 4 convolutional layers** 
- We're applying global average pooling to simplify feature maps
- The classifier contains:
  - Two dense layers (256→128 units)
  - BatchNormalization to stabilize training
  - Dropout (0.5) to prevent overfitting
  - Standard softmax output for classification
- We use SGD with momentum (0.9) and a lower learning rate (0.0005)
- We are training for 15 epochs to let fine-tuning work its magic

#### Why We're Selectively Freezing Layers

The idea of only making 4 layers trainable is for the below reasons:

The first layers in VGG16 are already great at finding edges, textures, and basic shapes - these work for almost any image task, including Lego bricks according to a bit of research on Google. The deeper layers are seemingly more specialized for ImageNet objects (dogs, cats, cars), which aren't very "Lego-like."

With this, we steer to model to:
- **Stop overfitting in its tracks**: Our Lego dataset isn't huge, so training all the layers is probably slighly unreasonable.

- **Save time and resources**: Training fewer parameters means faster iterations and less computing power needed.

- **Get the best of both worlds**: We keep all the powerful feature detection VGG16 learned from millions of ImageNet images while adapting it to our specific Lego task.

- **Gradient stability**: Prevents "catastrophic forgetting" of useful features and helps avoid vanishing/exploding gradients through the deep network.

Our configuration provides flexibility through the `trainable_layers` parameter, which can be adjusted based on dataset size and similarity to ImageNet classes.

### Evaluation Methodology
- Models are evaluated using a proper three-way split:
  - 80% training data for model training
  - 10% validation data for hyperparameter tuning and early stopping
  - 10% test data for final performance evaluation only
- This prevents data leakage between validation and test sets, giving a more realistic estimate of model performance
- Performance metrics include accuracy, precision, recall, and F1-score
- Learning curves show training and validation accuracy/loss over epochs
- Comparative analysis examines whether transfer learning improves performance


## Results - NEEDS UPDATE
The experiment yielded striking differences between our two approaches. Here's what we found:

### The Custom CNN's Struggles
Our simple CNN really struggled with the Lego classification task, achieving only about 14% accuracy after 15 epochs of training. Most Lego brick types were barely recognized or not recognized at all - for example, it completely failed to identify Brick_1x1, Brick_2x2, and several other classes with 0% recall. The best it could manage was about 78% recall on Plate_1x1_Slope pieces, but with terrible precision (only 8.5%).

The learning curves show the model was slow to improve, starting around 6% accuracy and crawling up to 16% by the end of training. Even after 15 epochs, there were no signs of overfitting - instead, the model was clearly underfitting the data.

### VGG16's Superior Performance
In stark contrast, the VGG16 transfer learning approach knocked it out of the park with 63% overall accuracy after just 5 epochs. Some classes were recognized with amazing reliability - Plate_1x1_Round had nearly perfect results with 99% F1-score, and several other classes achieved F1-scores above 70%.

What's particularly impressive is how quickly the VGG16 model learned. Starting at just 10% accuracy, it jumped to 45% on the training set and 63% on validation after only 5 epochs. The validation accuracy was still climbing in the final epoch, suggesting we could get even better results with more training time.

### Why Transfer Learning Won
The dramatic difference (14% vs 63% accuracy) clearly shows the power of transfer learning for this task. By leveraging VGG16's pre-trained weights from ImageNet, we inherited powerful feature extractors that could recognize edges, textures, and shapes relevant to our Lego classification problem.

This 4.5× improvement in accuracy is especially notable considering the VGG16 model trained for only 5 epochs compared to the CNN's 15 epochs. 

### Learning Dynamics Comparison
The custom CNN showed slow and limited improvement, with accuracy plateauing below 16% despite 15 epochs of training. The learning curves indicate persistent underfitting and low model capacity. In contrast, the VGG16 model achieved rapid and consistent gains, reaching over 60% validation accuracy in just 5 epochs. The validation accuracy consistently exceeded training, highlighting strong generalization from the pretrained features. Overall, the learning dynamics clearly favor the transfer learning approach.

### Further Improvements
The VGG16 model's learning trajectory suggests we'd likely see additional gains with more training epochs, as validation accuracy was still increasing in the final epoch. We might also benefit from fine-tuning some of the later convolutional layers in VGG16, which could adapt the generic ImageNet features more specifically to Lego brick characteristics.

### Parameter Analysis and Impact

**CNN Configuration Impact:**
- The relatively high learning rate (0.001) with limited training (15 epochs) likely contributed to the CNN's poor performance, as complex image classification tasks typically need more time to converge
- Our simple 3-layer CNN architecture with [32, 64, 128] filters was insufficient for capturing the nuanced features of Lego bricks
- The larger dense layers [512, 256] may have introduced too many parameters for the model to learn effectively from our limited dataset

**VGG16 Transfer Learning Advantages:**
- The lower learning rate (0.0001) paired with frozen convolutional layers allowed for stable adaptation of the pre-trained features
- Using average pooling effectively reduced dimensionality while preserving spatial information
- The simplified classifier (single 128-unit dense layer) prevented overfitting while leveraging VGG16's robust feature extraction
- Setting trainable_layers=0 (complete feature freezing) proved highly effective, demonstrating that ImageNet features transfer well to Lego classification

Interestingly, the VGG16 model achieved superior results despite using fewer dense layers and training for only 5 epochs compared to the CNN's 15 epochs, showing that transfer learning is efficient for this task.

## Requirements
- TensorFlow 2.x
- NumPy
- Matplotlib
- scikit-learn
- Pydantic