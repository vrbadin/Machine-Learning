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

## Chapter 6. Best Practices for Model Evaluation and Hyperparameter Tuning

### 6.1. Pipeline

`Pipeline` from `sklearn.pipeline` gives us a handy tool to stack several algorithms on top of each other, as long as they have `fit` and `transform`. When `fit` is called on this object, it executes `fit` and `transform` of each method in the order.

#### Exercise
  - Use breast cancer dataset
  - Stack StandardScaler, PCA and LogisticRegression into a pipeline.
  
### 6.2. k-fold Cross-Validation for Model Performance

One standard way of tuning model is a *holdout method* - split the data into training, validation and test data sets. Use validation for improving the training and test at the end. The main drawback of this method is the sensitivity on how the data is split. 

*k-fold cross-validation method* remediates this issue. We split the data into `k` folds, where at each iteration `k-1` folds are used for model training and one fold is used for testing. This procedure is repeated `k` times so that we obtain `k` models and performance estimates. Estimated performance is then used as average of these `k` perfomances.

One thing to be mindful of is to preserve the original data scaling. For example, if you have to apply `StandardScaler`, then you have to do it again for each choice of training set.

The method for k-fold cross-validation is `StratifiedKFold` from `sklearn.cross_validation`. From the same import, we can make use of `cross_val_score` which is much easier to use. Useful argument is `n_jobs` which controls the number of CPU used (the algorithm is clearly parallelizable). If set `n_jobs=-1`, you make use of all available compute power. 

### 6.3. Confusion Matrix

Given an actual and predicted class, positives are what predicted class classified as positive (same for negatives). Correctly classified positives are True Positives (TP) and incorrectly classified positives are False Positives (FP). This is represented by a confusion matrix, and there is a method `confusion_matrix` in `sklearn.metrics`.

Several metrics from confusion matrix:
  - `ERR = (FP + FN) / (FP + FN + TP + TN)`
      - *Prediction error* measures the total incorrectly classified
  - `ACC = 1 - ERR`
      - *Accuracy* measures the total correctly classified
  - `FPR = FP / N = FP / (FP + TN)`
  - `TPR = TP / P = TP / (FN + TP)`
      - *False Positive Rate* and *True Positive Rate* are useful for imbalanced problems (for instance, for tumor detection we want low FPR to not concern patient)
  - `PRE = TP / (TP + FP)`
  - `REC = TPR`
  - `F1 = 2 * PRE * REC / (PRE + REC)`
      - *Precision* and *Recall* are metrics used in defining *F1-score*.

### 6.4. Receiver Operating Characteristic (ROC)

ROC graphs are useful tools for selecting models for classification based on their performance with respect to the false positive and true positive rates, which are computed by shifting the decision threshold of the classifier. The diagonal of ROC graph can be interpreted as random guessing, and the perfect classifier would fall into the top left corner. Based on the ROC curve, we can then compute the *Area Under Curve (AUC)* to characterize the performance of a classification model.

#### Exercise
  - Use breast cancer dataset
  - Stack StandardScaler, PCA and LogisticRegression into a pipeline
  - Stratified k fold in 3 folds
  - Plot ROC for all 3 folds
  - Compute AUC and Accuracy directly
  
## Chapter 7. Combining Different Models for Ensemble Learning

The goal behind *ensemble methods* is to combine different classifiers into a meta-classifier that has a better generalisation performance than each individual classifier alone. The simplest one is based on a *majority voting principle*. 

### 7.1. Majority Voting

Same idea can be applied to multi-class and to binary voting, and we will further on assume binary setting. The idea is simple - take several classifiers and choose what majority has picked. The motivation behind this is following - assuming each classifier has the same, uncorrelated error `eps`, the error of majority voting ensemble will be based on a binomial distribution `\sum (n \over k) eps^k (1 - eps)^(1-k)`, which is smaller.

Furthermore, the voting itself can be weighted, so the formulation of the weighted majority vote is as follows:
`y = \argmax(i) \sum w_j Indicator(C_j = i)`, 
whereas for simple weights this simplifies into
`y = mode {C_1(x), ... , C_m(x)}`.
This can be conveniently translated into Python via
`np.argmax(np.bincount([0, 0, 1], weights=[0.2, 0.2, 0.6]))`,
and in the future there will also be `sklearn.ensemble.VotingClassifier`. The perfomance can then be improved via hyperparameter tuning of each classifier.

### 7.2. Bagging - building an ensemble of classifiers from bootstrap samples

Instead of using the same training set to fit the individual classifiers in the ensemble, we draw bootstrap samples (random samples with replacement) from the initial training set, which is why bagging is also known as *bootstrap aggregating*. Random forests are special case of bagging where we also use random feature subsets to fit individual decision trees.

We can invoke it via `sklearn.ensemble.BaggingClassifier`. Bagging is efficient in reducing the overfitting of decision trees.

### 7.3. Adaptive Boosting (AdaBoost)

The key concept behind boosting is to focus on training samples that are hard to classify, that is, to let the *weak learners* subsequently learn from misclassified training samples to improve the performance of the ensemble. In contrast to bagging, the initial formulation of boosting, the algorithm uses random subsets of training samples drawn without replacement. The original boosting procedure is summarised in four key steps:
  1. Draw a random subset of training samples `d_1` without replacement from the training set `D` to train a weak learner `C_1`.
  2. Draw a second random training subset `d_2` without replacement from the training set and add 50% of the samples that were previously misclassified to train a weak learner `C_2`.
  3. Find the training samples `d_3` in the training set `D` on which `C_1` and `C_2` disagree to train a third weak learner `C_3`.
  4. Combine the weak learners `C_1`, `C_2` and `C_3` via majority voting.
  
Boosting can lead to a decrease in bias as well as variance compared to bagging models. In practice, however, boosting algorithms such as AdaBoost are also known for their high variance (tendency to overfit).

In contrast to the original boosting procedure, AdaBoost uses the complete training set to train where the training samples are reweighted in each iteration to build a strong classifier that learns from the mistakes of the previous weak learners in the ensemble. The pseudocode is as follows (x denotes elementwise, * denotes scalar product):
  1. Set the weight vector `w` to uniform weights `\sum w_i = 1`
  2. for `j` in `m` boosting rounds, do the following:
  3. Train a weighted weak learner: `C_j = train(X, y, w)`
  4. Predict class labels: `hat(y) = predict(C_j, X)`
  5. Compute weighted error rate: `eps = w * (hat(y) == y)`
  6. Compute coefficient: `\alpha_j = 0.5 log ((1 - eps)/eps)`
  7. Update weights: `w = w x exp(-\alpha_j x hat(y) x y)`
  8. Normalize weights to sum to 1
  9. Compute final prediction: `hat(y) = sum(\alpha_j x predict(C_j, X)) > 0`
  
AdaBoost can be invoked via `sklearn.ensemble.AdaBoostClassifier`.

## Chapter 8. Applying Machine Learning to Sentiment Analysis

### 8.1. Bag-of-words Model

The idea behind bag-of-words model is quite simple:
  1. We create a *vocabulary* of unique *tokens* from the entire set of documents
  2. We construct a feature vector from each document that contains the counts of how often each word occurs in the particular document.

It can be constructed via `from sklearn.feature_extraction.text import CountVectorizer`. The count of occurences of a word in each document is called *raw term frequency*:
`tf(t, d)` - the number of times a term `t` occurs in a document `d`.

The sequence of items we talked about so far are called *1-gram* or *unigram* model - each item or token in the vocabulary represents a single word. More generally, the contiguous sequences of items is also called *n-gram*. Study has shown that n-grams of size 3 and 4 yield a good performance in anti-spam filtering of email messages. It can be controlled via a parameter `ngram_range` of `CountVectorizer`. 2-gram representation would be `ngram_range = (2, 2)`.

When we are analyzing text data, we often encounter words that occur across multiple documents from both classes. Those frequently occurring words typically don't contain useful or discriminatory information. Therefore, we can define *term frequency-inverse document frequency*:
`tf-idf = tf(t, d) * (idf(t, d) + 1)`, 
where `idf` is *inverse document frequency*:
`idf(t, d) = log ((1 + n_d) / (1 + df(d, t)))` and `n_d` is the total number of documents. We can invoke it via `from sklearn.feature_extraction.text import TfidfTransformer` which would by default normalise it in l2 sense (this is controlled by parameter `norm='l2'`).

#### Exercise
  - Define a simple set of sequences which would represent docs
  - Create a bag of words, then represent its dictionary and bag of words
  - Apply tf-idf on this set

### 8.2. Cleaning text data

In case we are reading an HTML document, we might want to remove tags, or we want to count smilies as words. One way of doing it is via regex expressions. Great tutorials on the topic are available on the Google Developers portal at https://developers.google.com/edu/python/regular-expressions or on the official Python's `re` module for regex at https://docs.python.org/3.4/library/re.html .

In the context of tokenization, another useful technique is is *word stemming*, which is the process of transforming a word into its root form that allows us to map related words to the same stem. The original algorithm is called `Porter stemmer algorithm`, which can be invoked via `from nltk.stem.porter import PorterStemmer`. Other popular stemming algorithms include the Snowball and Lancaster stemmers, both available in nltk library. Notice the deficiency of Porter stemmer - it would convert 'thus' into 'thu'. A way to avoid this is via *lemmatization* techniques, however it was observed to have little impact on final results.

Another useful technique is *stop-word removal*. Stop-words are simply those words that are extremely common in all sorts of texts and likely bear no useful information. 

#### Exercise
  - Test the difference between a regular tokenizer and Porter stemmer
  - Test the stop-word removal from nltk library
  
### 8.3. Further Algorithms

When dealing with large text documents, online algorithm instead of tfidf is available - `HashingVectorizer`.

A popular extension of the bag-of-words model is *Latent Dirichlet allocation*, which considers the latent semantic of the words.

A more modern alternative to bag-of-words is *word2vec*, an algorithm Google released in 2013. The word2vec is an unsupervised learning algorithm based on neural networks that attempts to automatically learn the relationships between words. The idea behind word2vec is to put words that have similar meaning into similar clusters; via clever vector-spacing, the model can reproduce certain words using simple vector math, for example, `king - man + woman = queen`.

## Chapter 10. Predicting Continuous Target Variables with Regression Analysis

### 10.1. Visualizing the important characteristics of a dataset

*Exploratory Data Analysis (EDA)* is an important and recommended first step prior to the training of a machine learning model. It may help us visually detect the presence of outliers, the distribution of the data, and the relationships between features.

Useful package is `seaborn`, which is a Python library for drawing statistical plots based on matplotlib. We will explore its scatterplot (`pairplot`) and heat map (`heatmap`) methods.

#### Exercise
  - Use housing dataset
  - Plot the scatterplot of 5 chosen features
    - Notice a linear relationship between RM and MEDV
    - Notice MEDV has a normal distribution, but contains several outliers
  - Plot a heatmap of a correlation matrix
    - The highest correlation has LSTAT, however from scatterplot we see the relationship is not linear
    - Therefore, we will regress MEDV on RM
    
### 10.2. Ordinary least squares linear regression model

It is invoked via `from sklearn.linear_model import LinearRegression`. The implementation makes use of the LIBLINEAR library and advanced optimization algorithms that work better with unstandardized variables.

#### Exercise
  - Use housing dataset and regress MEDV on RM
  - Plot the regression line and points
  
### 10.3. Fitting a robust regression model using RANSAC

As an alternative to throwing out outliers, we can use a robust method of regression *RANdom SAmple Consensus (RANSAC)* algorithm, which fits a regression model to a subset of the data, the so-called *inliers*.

The algorithm works as follows:
  1. Select a random number of samples to be inliers and fit the model.
  2. Test all other data points against the fitted model and add those points that fall within a user-given tolerance to the inliers.
  3. Refit the model using all inliers.
  4. Estimate the error of the fitted model versus the inliers.
  5. Terminate the algorithm if the performance meets a certain user-defined threshold or if a fixed number of iterations has been reached; go back to step 1 otherwise.
  
It is invoked via `from sklearn.linear_model import RANSACRegressor`. By default, scikit-learn uses *MAD (Median Absolute Deviation)* to select the inlier threshold.

#### Exercise
  - Use housing dataset and regress MEDV on RM
  - Plot the regression line and inlier vs outlier points

### 10.4. Evaluating the performance of linear regression models

When dealing with multiple linear regression, we can't always visualize the linear regression line, but we can plot residuals.  Those *residual plots* are commonly used graphical analysis for diagnosing regression models to detect nonlinearity and outliers, and to check if the errors are randomly distributed.

Another useful measure of model performance is *Mean Squared Error (MSE)*, which is simply the average value of the SSE cost function that we minimize to fit regression model. Sometimes you can report the *coefficient of determination (R^2)*, which can be understood as standardized version of MSE:
`R^2 = 1 - SSE/SST = 1 - MSE / Var(y)`.

#### Exercise
  - Use housing dataset and regress MEDV on all features
  - Plot the residuals, compute MSE and R^2 using their scikit-learn functions
  
### 10.5. Using regularized methods for regression

The most popular methods are:
  - *Ridge Regression* - add the squared sum of weights
  - *LASSO* - add the sum of absolute values. It can be used as a feature selection tool.
  - *Elastic Net* - take both sum of squared and absolute weights. Limitation of LASSO is that it selects at most `n` variables if `m>n`. L1 penalty generates sparcity and L2 overcomes some of the LASSO limitations.
  
They can be initialized as follows:
`from sklearn.linear_model import Ridge, Lasso, ElasticNet`
`ridge = Ridge(alpha=1.0)`
`lasso = Lasso(alpha=1.0)`
`elnet = ElasticNet(alpha=1.0, l1_ratio=0.5)`
For elastic net, if `l1_ratio=1.0`, it would be identical to LASSO.

### 10.6. Polynomial regression

We are able to use the same `LinearRegression` method with `from sklearn.preprocession import PolynomialFeatures` to perform polynomial regression. Simply define `cube = PolynomialFeatures(degree = 3)` which you then apply on the data via `cube.fit_transform(X)`.

### 10.7. Decision tree and random forest regression

Instead of using the Information Gain metric with the relative entropy measure (used in classification), we are using the Information Gain with MSE, so that we split the data into sum of squares which minimize the overall variance. 

As we have seen in classification, random forest usually has a better generalization performance than the individual decision tree due to randomness that help decrease the model variance. Other advantages of random forests are that they are less sensitive to the outliers in the dataset and don't require much parameter tuning. The only parameter in random forests that we typically need to experiment with is the number of trees in the ensemble.

#### Exercise
  - Regress MEDV on LSTAT using `from sklearn.tree import DecisionTreeRegressor`
  - Plot the decision tree line and note it is piecewise constant
  - Regress MEDV on all features using `from sklearn.ensemble import RandomForestRegressor`
  - Split the training vs test set and test for MSE, R^2 and residual plot
  
## Chapter 11. Working with unlabeled data - Clustering Analysis

In real-world applications of clustering, we do not have any ground truth category information about the samples; otherwise it would fall into the category of supervised learning. Our goal is to group the samples based on their feature similarities.

We will discuss several categories of clustering:
  - *prototype-based* - each cluster is represented by a prototype, which can either be centroid or the medoid (the most representative or frequently occurring point)
  - *hierarchical*
  - *density-based*

### 11.1. K-means and K-means++

K-means is a prototype-based algorithm. While it is very good at identifying clusters of spherical shape, one of the drawbacks is that we have to specify the number of clusters `k` a priori. It follows the steps:
  1. Randomly pick `k` centroids from the sample points as initial cluster centers
  2. Assign each sample to the nearest centroid `m_i`
  3. Move the centroids to the center of the samples that were assigned to it
  4. Repeat steps 2 and 3 until the cluster assignments do not change or a user-defined tolerance or a maximum number of iterations is reached

The question is how do we define similarity measure - one way to do it is to define it as the opposite of the distance. Commonly used is Euclidian distance, so we can describe k-means as an iterative approach of minimizing the within-cluster sum of squared errors (SSE), which is also called *cluster intertia*.

Random initialization can sometimes result in bad clusterings or slow convergence. The strategy to remediate is to place the initial cetroids far away from each other via *k-means++* algorithm, which leads to better and more consistent results than the classical k-means. The initialization can be summarized as follows:
  1. Initialize a empty set `M` to store the `k` centroids being selected
  2. Randomly choose the first centroid `m_j` from the input samples and assign it to `M`
  3. For each sample `x_i` not in `M`, find the minimum squared distance `d(x_i, M)^2` to any of the centroids `M`
  4. To randomly select the next centroid `m_p`, use weighted probability distribution equal to `d(m_p, M)^2 / (sum d(x_i, M)^2)`
  5. Repeat steps 2 and 3 until `k` centroids are chosen
  6. Proceed with the classical k-means

To use k-means++ with scikit-learn's `KMeans` object, we just need to set the `init` parameter to `k-means++` (the default setting) instead of `random`.

Another problem with k-means is that one or more clusters can be empty. 

### 11.2. Hard vs Soft clustering

*Hard clustering* describes a family of algorithms where each sample in a dataset is assigned to exactly one cluster, as in the k-means algorithm. In contrast, algorithms for *soft clustering* (also called *fuzzy clustering*) assign a sample to one or more clusters. A popular example is the *fuzzy C-means (FCM)* - also called *soft k-means*. The steps it follows are:
  1. Specify the number of `k` centroids and randomly assign the cluster memberships for each point
  2. Compute the cluster centroids `m_j`
  3. Update the cluster memberships for each point
  4. Repeat steps 2 and 3 until the membership coefficients don't change or tolerance or max iterations is reached.

The membership indicator is not binary as in k-means - it is based on `w_(i, j) = 1 / (sum_p ( |x-m_i| / |x-m_p| )^(2/(m-1)))`, and the weight would be `w^m`, where `m` is fuzziness coefficient and it is typically chosen to be 2. 

The center `m_j` is computed as the mean of all points with respect to its weight. FCM requires fewer iterations to reach converges than k-means, but it is slower. However, they tend to produce similar outputs.

### 11.3. Elbow method and silhouette plots

One of the challenges is identifying the number of clusters in the dataset. We will explore 2 methods - elbow method and silhouette plots.

Observe the plot of within cluster SSE for various values of number of clusters `k` - intuitively, as `k` increases, the SSE decreases. The idea behind *elbow method* is to identify the value of `k` where the distortion begins to increase most rapidly. In k-means, we have readily available value of within cluster SSE via `inertia_`.

Another intrinsic metric to evaluate the quality of a clustering is *silhouette analysis*, which can also be applied to clustering algorithms other than k-means. Silhouette analysis can be used as a graphical tool to plot a measure of how tightly grouped the samples in the clusters are. To calculate the *silhouette coefficient*, we follow these steps:
  1. Calculate the *cluster cohesion* `a^(i)` as the average distance between a sample `x^(i)` and all other points in the same cluster.
  2. Calculate the *cluster separation* `b^(i)` from the next closest cluster as the average distance between the sample `x^(i)` and all samples in the nearest cluster.
  3. Calculate the *silhouette* `s^(i) = ( b^(i) - a^(i) ) / max( b^(i), a^(i) )`.
  
The silhouette coefficient is bounded in the range -1 to 1. Ideal value is 1, obtained when `b^(i) >> a^(i)`. The silhouette coefficient is available as `silhouette_samples` from scikit-learn's `metric` module, and optionally `silhouette_scores` can be imported (which calculates the mean across all silhouette coefficients). As usual, we can specify the metric via `metric` parameter.
