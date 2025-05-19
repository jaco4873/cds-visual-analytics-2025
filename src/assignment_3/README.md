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
- Data is split with 80/10/10 % train, test and validation.

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
  - 10% validation data for early stopping
  - 10% test data for final performance evaluation only
- This prevents data leakage between validation and test sets, giving a more realistic estimate of model performance
- Performance metrics include accuracy, precision, recall, and F1-score
- Learning curves show training and validation accuracy/loss over epochs
- Comparative analysis examines whether transfer learning improves performance


## Results

Our experiments with two different approaches to Lego brick classification yielded valuable insights about the benefits of transfer learning in computer vision tasks, following the evaluation methodology outlined above.

### Overall Performance Comparison

The CNN model achieved respectable performance with a test accuracy of 82.69% after 15 epochs of training. Starting from around 12% accuracy, the model showed consistent improvement throughout training, with validation accuracy reaching 81.92% by the final epoch - closely tracking the training accuracy of 84.77%. This alignment between training and validation metrics indicates the model generalized well to unseen data within our 80-10-10 split.

The VGG16 transfer learning approach delivered superior performance, reaching 95.51% test accuracy - approximately a 13 percentage point improvement over the custom CNN. Interestingly, the VGG16 model showed rapid initial learning, with validation accuracy (94.42%) actually exceeding training accuracy (91.29%) in the final epoch, strongly suggesting that the pre-trained features transferred effectively to our Lego classification task.

### Learning Comparison

The custom CNN showed a gradual, steady learning curve with both training and validation metrics improving consistently across epochs. The narrow gap between final training (84.77%) and validation (81.92%) accuracies suggests the model found a good balance between fitting the training data and generalizing to the validation set.

The VGG16 model exhibited more dynamic learning behavior, with some fluctuations in validation accuracy but an overall stronger trajectory. By epoch 8, it had already achieved validation accuracy above 93%. The final validation accuracy of 94.42% actually exceeds the training accuracy of 91.29%, and this pattern continues with the test accuracy of 95.51% - a sign of excellent generalization across all three data splits.

The big difference in performance between the two models shows why transfer learning works so well for specific image tasks like this one. By using VGG16's pre-trained weights from ImageNet, our model started with ready-made feature detectors that could already spot important patterns in Lego bricks - things like edges, colors, textures, and shapes. By fine-tuning just the last 4 convolutional layers, we let the model adapt these general features to the specific characteristics of Lego bricks.

What's really interesting is that the VGG16 model not only got higher accuracy but also did better at generalizing to new images, as shown by its validation (94.42%) and test (95.51%) results being better than its training results (91.29%). This suggests that the patterns learned from the diverse ImageNet dataset work surprisingly well even for something as different as Lego brick classification.

### Architecture Impact

Our configuration choices for both models played significant roles in their performance. The CNN's architecture with three convolutional layers (increasing from 32 to 64 to 128 filters) and two dense layers (512, 256 units) provided sufficient capacity to learn meaningful features from scratch, but couldn't match the VGG16's pre-trained feature hierarchy.

For the VGG16 model, our strategy of freezing early layers while fine-tuning the last 4 convolutional layers proved highly effective. This approach preserved general visual feature detectors while allowing adaptation to Lego-specific characteristics. The classifier head we added - with two dense layers (256, 128 units) and a dropout rate of 0.5 - prevented overfitting despite the model's considerable capacity.

The performance difference between the models underscores that transfer learning from large, diverse datasets can dramatically improve results on specialized tasks with relatively small datasets, even when the target domain (Lego bricks) differs significantly from the source domain (ImageNet's natural images).

### Limitations

We did not perform hyperparameter optimization for either model, instead using fixed configurations based on common practices. The training process runs for a predetermined number of epochs (15) without implementing early stopping based on validation performance. This means models may train longer than necessary, or potentially benefit from additional training epochs. Additionally, we limited our exploration to specific architectures: a basic CNN and VGG16 transfer learning, without comparing against other model architectures or transfer learning approaches. Different data augmentation strategies or more extensive fine-tuning of the VGG16 layers might yield even better results. These limitations present opportunities for future work to further improve performance on the Lego classification task.

## Requirements
- TensorFlow 2.x
- NumPy
- Matplotlib
- scikit-learn
- Pydantic