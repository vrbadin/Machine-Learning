from chapter_3.plot_decision_regions import plot_decision_regions
import matplotlib.pyplot as plt

########### TAKEN FROM decision_trees.py ################
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
##########################################################

# 4. Perform PCA
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
lr  = LogisticRegression()
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca  = pca.transform(X_test_std)
lr.fit(X_train_pca, y_train)
plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.show()

# 5. Test on test data
plot_decision_regions(X_test_pca, y_test,classifier=lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend('lower left')
plt.show()

# 6. Explained variance
pca.explained_variance_ratio_