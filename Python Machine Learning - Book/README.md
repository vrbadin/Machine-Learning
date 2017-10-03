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

### 3.3. Support Vector Machines (SVMs)

In SVMs, the objective is to maximize *margin*. Margin is defined as the distance between the separating hyperplane (decision boundary) and the training samples that are closest to this hyperplane, which are the so-called *support vectors*.

The slack variable `ksi` is introduced for linearly non-separable data. The problem transforms into:
`w^T x >= 1 - ksi` if `y = 1`
`w^T x <= -1 + ksi` if `y = 0`,
so the objective to be minimized is `0.5*L2_sum(weights) + C sum(ksi)`.

Typically it has very similar performance to the logistic regression in practical classification tasks. Logistic regression has the advantage of being a simpler model, and can be easily updated, which is attractive when working with streaming data.

#### Exercise
- Use the same Iris dataset
- Use SVM with `C=1.0`.

### 3.4. Alternative Implementations in Scikit-learn

The  `Perceptron` and `LogisticRegression` classes used so far make use of the hihgly optimised LIBLINEAR library in C/C++. Likewise, `SVC` class makes use of LIBSVM, which is a specialised C/C++ library. The advantage of using these over Python implementations is that they allow for quick training of large amounts of linear classifiers. If the data is too large to fit into computer memory, we can use `SGDClassifier` class, which also supports online learning via `partial_fit` method. 

We can initialise perceptron, logistic regression and SVM in the following way:
`from sklearn.linear_model import SGDClassifier`
`ppn = SGDClassifier(loss='perceptron')`
`lr  = SGDClassifier(loss='log')`
`svm = SGDClassifier(loss='hinge')`

### 3.5. Kernel SVMs

The popularity of SVMs is due to its capability of being easily *kernelized* to solve nonlinear classification problems. One of the most widely used kernels is the *Radial Basis Function kernel (RBF kernel)* or Gaussian kernel:
`k(x^1, x^2) = exp{ -gamma L2_diff(x^1, x^2) }`, where `gamma` is the free parameter to be optimized.

Roughly speaking, the term kernel can be interpreted as a similarity function between a pair of samples. Very similar terms will result in 1, and very dissimilar samples in 0 (due to exponential term).

#### Exercise
- Use the same Iris dataset
- Use kernel SVM with `C=1.0`, `gamma=0.2` - soft boundary.
- Use kernel SVM with `C=1.0`, `gamma=100.0` - hard boundary.

### 3.6. Decision Trees

Decision trees break the feature space into rectangles. Popular information gain functions are Gini Impurity (`p(1 - p)`) and Entropy (`p log_2 p`), though they often yield similar results due to the similar shape.

We are able to store the decision tree in `.dot ` and view it in Graphviz (program to download) via the following command:
`dot -Tpng tree.dot -o tree.png`

#### Exercise
- Use the same Iris dataset
- Use decision tree with `max_depth=3` and entropy function.

### 3.7. Random Forests

The random forest algorithm can be summarised in the following steps:
1. Draw a random *bootstrap* sample of size `n` (randomly choose `n` samples from the training set with replacement).
2. Grow a decision tree - at each node:
  - Randomly select `d` features without replacement.
  - Split the node using the feature that provides the best split according to the objective function.
3. Repeat 1-2. `k` times.
4. Aggregate the prediction by each tree to assign the class label by *majority vote*.

Defaults used are:
- `n` - number of samples in the training set, which usually provides good bias-variance trade-off,
- `d = sqrt(m)` - where `m` is the number of features,
- `n_jobs` - argument which allows for parallelisation.

#### Exercise
- Use the same Iris dataset
- Use random forest with `n=10` and entropy function.

### 3.8. k-Nearest Neighbors (KNN)

The steps KNN follows are:
1. Choose the number or `k` and distance metric
2. Find the `k` nearest neighbours of the sample that we want to classify
3. Assign the class label by majority vote

Storage space can become a challenge if we are working with the large datasets. It is the memory-based algorithm - there is no training step involved. There are no parameters to be optimised.

#### Exercise
- Use the same Iris dataset
- Use KNN with `k=5` and Minkowski metric with `p=2` (Euclid space).

## Chapter 4. Data Preprocessing

### 4.1. Dealing with missing data

  - There is a `dropna` method which allows to drop particular rows/columns which contain NAs. Additionally, you can drop in a thresholded way (i.e. rows that contain more than 4 NAs).
  - Instead of dropping NAs, you can *impute the data*. `Imputer` class is used, with its parameter `strategy` allowing to specify the method of imputation. Some of the most common strategies are `mean`, `median` and `most_frequent`.
 
### 4.2. Handling categorical data
  - Labels should be mapped to int.
  - `LabelEncoder` allows for mapping string to int. For example, it maps `{red, green, blue}` to `{0, 1, 2}` and that is the issue - it introduces ordering.
  - `OneHotEncoder` mitigates the issue above and creates a new boolean columns.
  - `DataFrame` has a very convenient method for OneHotEncoder - `get_dummies`.
  
### 4.3. Partitioning a dataset in training and test sets
  - `train_test_split` from `cross_validation` module handles the split. 
  - Most common splits are 60:40, 70:30, 80:20, however for the large datasets also common ones are 90:10 and 99:1.
  
### 4.4. Bringing features onto the same scale
  - `MinMaxScaler` scales proportionally from 0 to 1
  - `StandardScaler` scales by mean and variance
  
### 4.5. Selecting meaningful features
  - There are 2 ways to reduce dimensionality: *feature selection* and *feature extraction*. We are dealing currently with the feature extraction.
  - Classical feature selection algorithm is *Sequential Backward Selection*, which aims to reduce dimensionality of the initial feature subspace with a minimum decay in performance of the classifier. We implemented `SBS` class which takes a classifier as input (for instance, KNN Classifier).
  - Using random forest, we can measure feature importance as the averaged impurity decrease computed from all decision trees in the forest without making any assumptions whether our data is linearly separable or not. The issue could appear with highly correlated features - random forest may prefer much more one of them.
  
## Chapter 5. Compressing Data via Dimensionality Reduction

### 5.1. Principal component analysis (PCA)

PCA directions are highly sensitive to data scaling, and we need to standardize the features prior to PCA if the features were measured on different scales and we want to assign equal importance to all features.

#### Exercise
  - Use the iris dataset and standardised data as before
  - Use the plot_decision_regions
  - Use the logistic regression classifier on data using the first 2 principal components
  
### 5.2. Supervised data compression via linear discriminant analysis (LDA)

The general concept behind LDA is very similar to PCA, whereas PCA attempts to find the orthogonal component axes of maximum variance in a dataset; the goal in LDA is to find the feature subspace that optimizes class separability. 

Thus, we might intuitively think that LDA is a superior feature extraction technique for classification tasks than PCA. However, it was reported that preprocessing via PCA tends to result in better classification results in an image recognition tasks in certain cases, for instance, if each class consists of only a small number of samples.

One assumption in LDA is that the data is *normally distributed*. Also, we assume that the *classes have identical covariance matrices* and that *the features are statistically independent of each other*. However, even if one or more of those assumptions are slightly violated, LDA for dimensionality reduction can still work.

Steps of LDA approach:
  1. Standardise the `d`-dimensional dataset 
  2. For each class, compute the `d`-dimensional mean vector
  3. Construct the between-class scatter matrix `S_B` and the within-class scatter matrix `S_w`
  4. Compute the eigenvectors and corresponding eigenvalues of the matrix `S_w^{-1} S_B`
  5. Choose the `k` eigenvectors that correspond to the `k` largest eigenvalues to construct a `d \times k`-dimensional transformational matrix `W`; the eigenvectors are the columns of this matrix
  6. Project the samples onto the new feature subspace using the transformation matrix `W`




