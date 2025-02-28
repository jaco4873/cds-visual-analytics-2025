# Assignment 1: Building a Simple Image Search Algorithm

## Overview
For this assignment, you'll be using OpenCV to design a simple image search algorithm. The dataset is a collection of over 1000 images of flowers, sampled from 17 different species. The dataset comes from the Visual Geometry Group at the University of Oxford. Full details of the data can be found [here](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html).

## Instructions

Your task is to write code that performs the following steps:

1. Define a particular image that you want to work with
2. For that image:
   - Extract the colour histogram using OpenCV
   - Extract colour histograms for all of the *other* images in the dataset
   - Compare the histogram of your chosen image to all of the other histograms
     - Use the `cv2.compareHist()` function with the `cv2.HISTCMP_CHISQR` metric
3. Find the five images which are most similar to the target image
4. Save a CSV file to the folder called `out`, showing the five most similar images and the distance metric:

| Filename | Distance |
|----------|----------|
| target   | 0.0      |
| filename1| ---      |
| filename2| ---      |
| ...      | ...      |

## Objective

This assignment is designed to test that you can:
- Work with larger datasets of images
- Extract structured information from image data using OpenCV
- Quantitatively compare images based on these features, performing distant viewing

## Notes

- A zip file containing data can be found at the link above, or under the content tab.
- You'll need to first unzip the flowers before you can use the data!

## Submission Guidelines

- Your code should include functions that you have written wherever possible.
- Try to break your code down into smaller self-contained parts, rather than having it as one long set of instructions.
- You are welcome to submit your code either as a Jupyter Notebook or as a .py script.
- If you do not know how to write .py scripts, don't worry - we're working towards that!