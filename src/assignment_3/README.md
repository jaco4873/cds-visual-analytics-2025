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

You can also run the code without run.sh:
```bash
# Navigate to src from the project root
cd src

# Run as module
uv run python -m assignment_3.main 

# With custom arguments
uv run python -m assignment_3.main --data-dir /path/to/lego/data --output-dir /path/to/output
```

## Configuration

The project uses a central configuration system in `config.py` that can be modified to customize various aspects:

- Data Configuration: Control image dimensions and data directory paths settings
- CNN Model Configuration: Adjust architecture parameters, learning rates, and training settings
- VGG16 Model Configuration: Configure transfer learning parameters, fine-tuning depth, and training settings
- Output Configuration: Set output directory paths for saving models, reports, and visualizations

All settings are managed through Pydantic classes, making it easy to modify behavior without changing code.

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
        ├── classification_report.txt
        ├── learning_curves.png
        └── training_history.json
```

## Dataset
- Source and format: The dataset consists of images of various Lego bricks arranged in folders named after the brick type. 
- The data should be placed in the `data/lego` directory. 
- We use the data from the "Cropped Images" folder, which contains images with backgrounds removed.

## Implementation Details

### Data Processing
- Images are resized to 224×224 pixels with 3 color channels
- Batches of 32 images are processed at a time
- Data is split with 80/10/10 % train, test and validation.
- We experimented with various data augmentation techniques (rotation, zoom, flip, shift) but found that models achieved better validation results without any augmentation. The final implementation therefore uses the original images without augmentation.

### CNN Model Architecture
The custom CNN is configured with the following architecture:
- Input shape: (224, 224, 3)
- Convolutional layers with increasing filter complexity: [32, 64, 128]
- 3×3 kernel sizes with 2×2 max pooling
- Two dense layers [512, 256] with ReLU activation
- Dropout (0.5) for regularization
- Softmax output layer for classification
- Adam optimizer with learning rate 0.001
- Early stopping with patience=8 to prevent overfitting
- Trained for up to 30 epochs (may stop earlier with early stopping)

### VGG16 Transfer Learning Model
- We're using VGG16 (pretrained on ImageNet) as our foundation
- We've removed the top classification layers (include_top=False)
- We're fine-tuning the last 4 convolutional layers 
- We're applying global average pooling to simplify feature maps
- The classifier contains:
  - Two dense layers (256→128 units)
  - BatchNormalization to stabilize training
  - Dropout (0.5) to prevent overfitting
  - Standard softmax output for classification
- We use SGD with momentum (0.9) and a lower learning rate (0.0005)
- Early stopping with patience=8 to prevent overfitting
- We are training for up to 30 epochs (may stop earlier with early stopping)

#### Transfer Learning Strategy

Our implementation freezes earlier VGG16 layers while making only the final 4 convolutional layers trainable. This approach makes use of the hierarchical nature of convolutional networks, where initial layers capture universal visual primitives (edges, textures, basic shapes) that generalize well across domains, while deeper layers represent increasingly task-specific features. 

First, it addresses the potential for overfitting given our relatively constrained dataset size. By limiting the number of trainable parameters, we create a more favorable ratio between trainable weights and available training examples.

Second, this approach significantly reduces computational requirements compared to full fine-tuning.

Third, the method creates an balance between knowledge transfer and domain adaptation. By preserving the robust feature extractors from VGG16's early layers (trained on millions of ImageNet images) while allowing later layers to adapt to Lego-specific characteristics,

Finally, selective freezing provides gradient stability benefits during training. By reducing the network's effective depth from the perspective of backpropagation, we mitigate vanishing gradient issues and prevent catastrophic forgetting of useful pre-trained features.

The transfer learning configuration is parameterized through the `trainable_layers` setting, allowing experimentation with freezing strategies based on dataset characteristics and target domain similarity to the ImageNet source distribution.

### Evaluation Methodology

Our evaluation framework implements a three-way data partitioning, allocating 80% of available data for model training, 10% for validation-based early stopping decision, and 10% for unbiased final performance assessment. This approach prevents potential data leakage between model selection and evaluation phases, giving a more realistic estimate of model performance.

Performance quantification incorporates both primary metrics (accuracy) and secondary distributional metrics (precision, recall, F1-score) to provide insight into classification behavior across all class categories. Temporal learning dynamics are visualized through epoch-wise training and validation metrics, enabling identification of potential optimization issues including underfitting, overfitting, and convergence patterns.

## Results

Our experiments with two different approaches to Lego brick classification yielded valuable insights about the benefits of transfer learning in computer vision tasks, following the evaluation methodology outlined above.

### Overall Performance Comparison

The CNN model achieved respectable performance with a test accuracy of 84.40% after 28 epochs of training. Starting from around 13% accuracy, the model showed consistent improvement throughout training, with validation accuracy reaching 86.16% by epoch 25 (the best validation epoch). The training accuracy continued to increase to 95.58% by the final epoch, suggesting some potential overfitting despite early stopping being applied. The model seems to have generalized well to unseen data within our 80-10-10 split.

The VGG16 transfer learning approach delivered superior performance, reaching 95.51% test accuracy - approximately an 11 percentage point improvement over the custom CNN. Interestingly, the VGG16 model showed rapid initial learning, with validation accuracy (94.42%) actually exceeding training accuracy (91.29%) in the final epoch, strongly suggesting that the pre-trained features transferred effectively to our Lego classification task. The best validation accuracy of 96.65% was achieved in epoch 12.

### Learning Comparison

The custom CNN showed a somewhat uneven learning curve with fluctuations in validation metrics, though with an overall positive trajectory. The training accuracy steadily increased to 95.58% by the final epoch, while validation accuracy peaked at 86.16% in epoch 25 before declining slightly, triggering early stopping. This growing gap between training and validation metrics indicates the model was beginning to overfit to the training data.

The VGG16 model exhibited more dynamic learning behavior, with some fluctuations in validation accuracy but an overall stronger trajectory. By epoch 8, it had already achieved validation accuracy above 93%. The final validation accuracy of 94.42% actually exceeds the training accuracy of 91.29%, and this pattern continues with the test accuracy of 95.51% - a sign of excellent generalization across all three data splits.

The big difference in performance between the two models shows why transfer learning works so well for specific image tasks like this one. By using VGG16's pre-trained weights from ImageNet, our model started with ready-made feature detectors that could already spot important patterns in Lego bricks - things like edges, colors, textures, and shapes. By fine-tuning just the last 4 convolutional layers, we let the model adapt these general features to the specific characteristics of Lego bricks.

What's really interesting is that the VGG16 model not only got higher accuracy but also did better at generalizing to new images, as shown by its validation peaks (96.65% in epoch 12) and test (95.51%) results being better than its training results (91.29%). This suggests that the patterns learned from the diverse ImageNet dataset work surprisingly well even for something as different as Lego brick classification.

### Limitations

We did not perform hyperparameter optimization for either model, instead using fixed configurations based on common practices. We implemented early stopping with a patience of 8 epochs to prevent overfitting. Additionally, we limited our exploration to specific architectures: a basic CNN and VGG16 transfer learning, without comparing against other model architectures or transfer learning approaches. 

Although our initial experiments with data augmentation showed better validation results without augmentation, further experimentation with different augmentation strategies (such as more targeted transformations specific to the Lego domain) might further improve results. Similarly, more extensive fine-tuning of the VGG16 layers might yield even better performance. These limitations present opportunities for future work to further improve performance on the Lego classification task.

## Requirements
- TensorFlow 2.x
- NumPy
- Matplotlib
- scikit-learn
- Pydantic