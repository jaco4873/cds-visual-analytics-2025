# Assignment 3: Transfer Learning with Pretrained CNNs

## Overview
This assignment implements two approaches for classifying Lego brick images:
1. A custom CNN trained directly on the image data
2. A transfer learning approach using VGG16 as a feature extractor

The goal is to compare these two approaches and determine whether transfer learning improves performance for this specific image classification task.

## Dataset
The dataset consists of images of various Lego bricks arranged in folders named after the brick type. The data should be placed in the `data/lego` directory. We use thee data from the "Cropped Images" folder, which contains images with backgrounds removed.

## Project Structure
```
src/assignment_3/
├── config.py                    # Configuration settings
├── main.py                      # Main entry point
├── assignment_description.md    # Assignment description
├── README.md                    # Project documentation
├── session9_inclass_rdkm.ipynb  # In-class notebook reference
├── services/                    # Service modules
│   ├── base_classifier_model.py # Base model class
│   ├── cnn_model.py             # Direct CNN classifier
│   ├── data_service.py          # Data loading and preprocessing
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


## Implementation Details

### Data Processing
- Images are resized to 224×224 pixels with 3 color channels
- Batches of 32 images are processed at a time
- Data is split with 20% for validation during training
- Image normalization scales pixel values to [0-1]
- Data augmentation is applied to improve model robustness:
  - Rotation (±20 degrees)
  - Zoom (±15%)
  - Width/height shifts (±20%)
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
The transfer learning approach follows the methodology taught in class:
- Uses VGG16 pretrained on ImageNet as the base model
- Removes the top classification layers (include_top=False)
- Freezes all convolutional layers to preserve learned features
- Applies global average pooling to reduce dimensionality
- Adds a custom classifier on top with:
  - A single dense layer with 128 units (following class example)
  - BatchNormalization for improved training stability
  - Dropout (0.5) for regularization
  - Softmax output layer for classification
- Uses Adam optimizer with lower learning rate (0.0001)
- Trained for 5 epochs to prevent overfitting

### Evaluation Methodology
- Models are evaluated using a two-way data split (training and validation)
- **Limitation**: The implementation reuses the validation set as the test set rather than using a true three-way separation, which may lead to slightly optimistic performance estimates
- Performance metrics include accuracy, precision, recall, and F1-score
- Learning curves show training and validation accuracy/loss over epochs
- Comparative analysis examines whether transfer learning improves performance

## How to Run

### Using the run.sh Script
The simplest way to run the assignment is using the provided run.sh script:
```bash
./run.sh
```
Then select option 3 from the menu.

### Manual Execution
You can also run the code directly:
```bash
# From the project root
uv run -m assignment_1.main 

# With custom arguments
uv run -m assignment_1.main  --data_dir /path/to/lego/data --output_dir /path/to/output
```

### Command Line Arguments
- `--data_dir`: Path to the Lego data directory (default: ./data/lego)
- `--output_dir`: Path to save the output (default: ./src/assignment_3/output)
- `--cnn_only`: Train only the CNN model
- `--vgg16_only`: Train only the VGG16 model

## Results
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

Interestingly, the VGG16 model achieved superior results despite using fewer dense layers and training for only 5 epochs compared to the CNN's 15 epochs, highlighting the efficiency of transfer learning for this task.

## Requirements
- TensorFlow 2.x
- NumPy
- Matplotlib
- scikit-learn
- Pydantic