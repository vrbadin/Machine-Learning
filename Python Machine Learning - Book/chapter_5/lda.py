# 1. Get the wine data
from data.wine import get_train_test
X_train_std, X_test_std, y_train, y_test = get_train_test()

# 2. Perform LDA
from sklearn.lda import LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)

# 3. Plot the decision regions
from sklearn.linear_model import LogisticRegression
from chapter_3.plot_decision_regions import plot_decision_regions
import matplotlib.pyplot as plt
lr = LogisticRegression()
lr = lr.fit(X_train_lda, y_train)
plot_decision_regions(X_train_lda, y_train, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.show()

# 4. Plot for test data
X_test_lda = lda.transform(X_test_std)
plot_decision_regions(X_test_lda, y_test, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.show()