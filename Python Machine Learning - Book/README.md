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

### 2.2. Adaline 
Adaline (ADAptive LInear NEuron) is another type of single-layer neural network. Weights are being updated based on a linear activation function rather than a unit step function like in the perceptron. A `quantizer` (similar to the unit step function) is then used to predict the class labels. We are using gradient descent in 2 versions:
- Batch gradient descent - uses all of the samples for the update
- Stochastic gradient descent - uses randomised (or online) samples one at the time for the update
It is also possible to do mini-batch which is combination of the two - use the `n` samples at the time for the update (`n=50` for instance)

#### Exercise

- Use batch gradient descent on Iris dataset, shows overshooting and small update step properties
- After normalising the sample input, results improve a lot
- Gradient descent shows steeper convergence
- Convergence and decision regions plotted

## Chapter 3. Machine Learning Classifiers Using Scikit-learn


*No Free Lunch theorem* - no single classifier works best across all possible scenarios.

Main steps that are involved in training a machine learning algorithm:
1. Selection of features.
2. Choosing a performance metric.
3. Choosing a classifier and optimization algorithm.
4. Evaluating the performance of the model.
5. Tuning the algorithm.

#### Exercise
- Iris data loading via scikit-learn
  - Flower names are already stored as integers (0, 1, 2), which is recommended for the optimal performance of many machine learning libraries.
- Using cross-validation to split the training and test data
- Using `StandardScaler` to preprocess the data
- Train perceptron model. Library automatical uses *One-vs.-Rest (OvR)* method, which allows us to feed the three flower classes all at once.



