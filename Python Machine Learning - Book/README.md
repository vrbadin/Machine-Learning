# Synopsis

The code closely follows the book *Python Machine Learning* by Sebastian Raschka. 

# Contents

## Chapter 2. Training Machine Learning Algorithms for Classification

### 2.1. Perceptrons 
Convergence is only guaranteed if the two classes are linearly separable and the learning rate is sufficiently small. If the two classes can't be separated by a linear decision boundary, we can set a maximum number of passes over the training set (*epochs*) and/or threshold for the number of tolerated misclassifications - the perceptron rule would never stop updating the weights otherwise.

#### Exercise

- Perceptron Interface - perceptron objects can learn from data via a `fit` method, and make predictions via a separate `predict` method
- Uses first 100 samples from Iris dataset, classifying based on 2 attributes - petal and sepal
- Perceptron converges after 6 epochs
- Convergence and decision regions plotted

## 2.2. Adaline 
Adaline (ADAptive LInear NEuron) is another type of single-layer neural network.


