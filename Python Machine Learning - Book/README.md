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

### 3.1. Perceptrons via Scikit-learn

#### Exercise
- Iris data loading via scikit-learn
  - Flower names are already stored as integers (0, 1, 2), which is recommended for the optimal performance of many machine learning libraries.
- Using cross-validation to split the training and test data
- Using `StandardScaler` to preprocess the data
- Train perceptron model. Library automatical uses *One-vs.-Rest (OvR)* method, which allows us to feed the three flower classes all at once.
- Plot decision region, highlighting the test data.

### 3.2. Logistic Regression

It performs very well on linearly separable sets. 

*Odds ratio* - given positive event `p`, odds ratio is defined as `p/(1-p)`
*Logit function* - log of odds ratio, transforms (0, 1) to the whole set of reals. Its inverse is sigmoid function.

In Adaline, activation function is an identity. In logistic regression, the activation function is sigmoid. The output of the sigmoid function is then interpreted as the probability of particular sample belonging to class 1.

Log-likelihood is sum of the terms `ylog(f(y)) + (1-y)log(f(1-y))`, which is either the first or the second term in the sum, depending on whether `y=0` or `y=1`.

The regularization is done by adding a term `0.5*lambda*L2_sum(weights)`. Convention is to instead deal with its inverse `C = 1/lambda`.

#### Exercise
- Use the same Iris dataset
- Perform logistic regression using `C=1000.0`.
- Perform logistic regressions for a set of values `C`, observing the shrinkage of parameters for low `C`.

### 3.2. Support Vector Machines (SVMs)

In SVMs, the objective is to maximize *margin*. Margin is defined as the distance between the separating hyperplane (decision boundary) and the training samples that are closest to this hyperplane, which are the so-called *support vectors*.

The slack variable `ksi` is introduced for linearly non-separable data. The problem transforms into:
`w^T x >= 1 - ksi` if `y = 1`
`w^T x <= -1 + ksi` if `y = 0`,
so the objective to be minimized is `0.5*L2_sum(weights) + C sum(ksi)`.

#### Exercise
- Use the same Iris dataset
- Use SVM with `C=1.0`.







