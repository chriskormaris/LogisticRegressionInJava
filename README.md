# LogisticRegressionInJava

by Chris Kormaris, August 2017

Programming Language: Java

This Logistic Regression implementation is linear, meaning it uses linearly separable data, if plotted on a 2D plane. 
It classifies the given data to two categories. The learning method is supervised, because labeled data are used during training.

Based on code from this repository: <a href="https://github.com/tpeng/logistic-regression">logistic-regression</a>

## Logistic Regression for Digit Classification of MNIST Dataset

First, unzip the compressed file *"mnisttxt.zip"* in the same directory where the file is already located.

The algorithm classifies the test data between two categories, as digit 1 or digit 7.

All the classifiers had the same accuracy on the test data, which was: **99.58 %**

## Logistic Regression for Spam-Ham Classification

First, unzip the compressed file *"LingspamDataset.zip"* in the same directory where the file is already located.

The algorithm classifies the test data between two categories, as "spam" or "ham".

The best accuracy achieved, without using regularization, was: **95 %**

The worst accuracy achieved, with regularization, was: **94.62 %** (the classifier did not need regularization)

#### Difference between Gradient Descent and Gradient Ascent
With gradient descent we aim to minimize a cost function, whereas with gradient ascent we aim to maximize a likelihood estimate. The results of the cost function and the likelihood estimate in each iteration should be the same, but with opposite signs.


The regularization term is used to avoid and reduce overfitting, in order to make our classifier unbiased towards a specific category.

