# Assignment 2 - Image Classification

## Instructions

### **Assignment 2 - Classification Benchmarks with Logistic Regression and Neural Networks**

For this assignment, we'll be writing scripts to classify the **CIFAR-10** dataset. You can read more about this dataset [here](https://www.cs.toronto.edu/~kriz/cifar.html).

### **Requirements**
You should write code that performs the following tasks:

1. **Load the CIFAR-10 dataset**  
2. **Preprocess the data** (e.g., convert to greyscale, normalize, reshape)  
3. **Train a classifier on the data**  
   - Implement **two classifiers**:  
     - A **Logistic Regression classifier**  
     - A **Neural Network classifier**  
4. **Save a classification report**  
5. **Save a plot of the loss curve during training**  
   - ⚠️ **Update**: This is only possible for the `MLPClassifier` in `scikit-learn`.  

You should write **two separate scripts**:
- One script for **Logistic Regression**
- One script for **Neural Network (MLPClassifier)**  

Both scripts should use **scikit-learn** to train and evaluate model performance.

---

## **Starter Code**

The CIFAR-10 dataset already includes a **train/test split** and can be loaded as follows:

```python
from tensorflow.keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
```

Once you have loaded the data, made it greyscale, and scaled the values, you will need to reshape the array to be the correct dimensions - essentially flattening the 2D array like we saw with greyscale histograms.

You can do that in the following way using numpy:

```python
X_train_scaled.reshape(-1, 1024)
X_test_scaled.reshape(-1, 1024)
```

## **Tips**

- Make sure to check the different parameters available with, for example, the `MLPClassifier()` in scikit-learn. Experiment with different model sizes and parameters.
- The training data comprises 50,000 examples - just be aware that the `MLPClassifier()` can be a little slow!
- The CIFAR-10 dataset you download does not have explicit label names but instead has numbers from 0-9. You'll need to make a list of labels based on the object names - you can find these on the website.
- You should structure your project by having scripts saved in a folder called `src`, and have a folder called `output` where you save the classification reports.

## **Purpose**

- To ensure that you can use scikit-learn to build simple benchmark classifiers on image classification data
- To demonstrate that you can build reproducible pipelines for machine learning projects
- To make sure that you can structure repos appropriately