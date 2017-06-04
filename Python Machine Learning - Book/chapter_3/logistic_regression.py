# 1. Load Iris using sklearn
from data.iris import get_skl
iris = get_skl()
X = iris.data[:, [2, 3]]
y = iris.target

# 2. Split into training and testing data randomly
#    30% test data / 70% training data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

# 3. Standardise the features (for SGD)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# 4. Perform the logistic regression
# THE DIFFERENT PART COMPARED TO PERCEPTRON
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)

# 5. Plot decision regions
from chapter_3.plot_decision_regions import plot_decision_regions
import matplotlib.pyplot as plt
import numpy as np
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std,
                      y=y_combined,
                      classifier=lr,
                      test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

# 6. Prediction probabilities
lr.predict_proba(X_test_std[0, :])

# 7. Perform regressions for a set of C's.
#    Observe shrinkage of params for low C (high lambda).
weights, params = [], []
for c in np.arange(-5, 5):
    lr = LogisticRegression(C=10.0**(c), random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10.0**(c))
weights = np.array(weights)
plt.plot(params, weights[:, 0],
         label='petal length')
plt.plot(params, weights[:, 1], linestyle='--',
         label='petal width')
plt.ylabel('weight coefficients')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log')
plt.show()