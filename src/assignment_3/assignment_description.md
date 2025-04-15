# Assignment 3 - Transfer Learning with Pretrained CNNs

## Overview
This assignment focuses on document classification using pretrained Convolutional Neural Networks (CNNs). We'll be comparing the performance of a CNN trained directly on image data versus using transfer learning with a pretrained CNN (VGG16).

## Background
Imagine you're building a Lego landscape and you're missing specific bricks. To order them, you need to know their exact names. This assignment involves developing a classifier that can identify different types of Lego bricks from images. The task is non-trivial as we need to classify based on shape rather than just color, making it an ideal candidate for exploring CNN and transfer learning approaches.

## Dataset
We'll be working with a Lego brick dataset that contains:
- Base images of Lego bricks
- Cropped images focusing on the Lego bricks with backgrounds mostly removed

The data is arranged into folders named after the label of the images they contain. It's available in the shared drive on UCloud.

## Requirements
You should write code that performs the following tasks:

1. **Load the Lego data**
   - Suggestion: Use `tf.keras.preprocessing.image_dataset_from_directory` for efficient loading

2. **Train a CNN classifier directly on the data**
   - Present a classification report for this model
   - Generate and save learning curves

3. **Train a CNN classifier using VGG16 as a feature extractor**
   - Present a classification report for this model
   - Generate and save learning curves
   
4. **Analysis**
   - Include a short description that compares the performance of both models
   - Specifically address whether the pretrained CNN (VGG16) improves performance

## Deliverables
- Python scripts (`.py` files)
- Classification reports saved in the `output` folder
- Learning curves saved in the `output` folder
- Analysis of results

## Project Structure
- Scripts should be saved in a folder called `src`
- Output files should be saved in a folder called `output`

## Tips
- The images are arranged into folders which have the label name for those images
- Using `tf.keras.preprocessing.image_dataset_from_directory` is recommended for loading and processing data
- Consider the computational requirements when training the models, especially with the full dataset

## Purpose
- To demonstrate proficiency in using TensorFlow to train Convolutional Neural Networks
- To create pipelines for transfer learning with pretrained CNNs
- To show understanding of how to interpret machine learning outputs in the context of supervised machine learning on image data