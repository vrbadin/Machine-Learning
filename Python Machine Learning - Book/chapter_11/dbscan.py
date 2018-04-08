import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Create a dataset - half moons
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200,
                  noise=0.05,
                  random_state=0)
plt.scatter(X[:, 0], X[:, 1])
plt.show()


# 2. Apply DBSCAN on this dataset
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.2,
            min_samples=5,
            metric='euclidean')
y_db = db.fit_predict(X)
plt.scatter(X[y_db==0, 0],
            X[y_db==0, 1],
            c='lightblue',
            marker='o',
            s=40,
            label='cluster 1')
plt.scatter(X[y_db==1, 0],
            X[y_db==1, 1],
            c='red',
            marker='s',
            s=40,
            label='cluster 2')
plt.legend()
plt.show()