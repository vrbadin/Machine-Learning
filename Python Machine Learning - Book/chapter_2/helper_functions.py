import numpy as np
import collections
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import data.iris as dataSource

def get_data():
    '''
    Iris dataset.
    First 50 correspond to Iris-Setosa,
    second 50 correspond to Iris-Versicolor.

    Features used are only 2:
    - sepal length,
    - petal length.

    Returns
    ---------
    Tuple containing X and y.
    '''
    rawData = dataSource.get()
    y = rawData.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', - 1, 1)
    X = rawData.iloc[0:100, [0, 2]].values

    Data = collections.namedtuple('Data', ['X', 'y'])
    res = Data(X=X, y=y)
    return res

def plot_input_data():
    '''
    Graphic representation of input data.
    '''
    data = get_data()
    plt.scatter(data.X[:50, 0], data.X[:50, 1],
                color='red', marker='o', label='setosa')
    plt.scatter(data.X[50:100, 0], data.X[50:100, 1],
                color='blue', marker='x', label='versicolor')
    plt.xlabel('sepal length')
    plt.ylabel('petal length')
    plt.legend(loc='upper left')
    plt.show()

def plot_decision_regions(X, y, classifier, resolution=0.02):
    '''
    Helper function for plotting decision regions.
    '''

    # setup marker generator and color map
    markers = ('s', 'x', 'o', 'm', 'v')
    colors  = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

def standardise(X):
    '''
    Transforms X using standardisation.
    '''
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    return X_std