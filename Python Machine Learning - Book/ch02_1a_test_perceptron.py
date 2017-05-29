import numpy as np
import matplotlib.pyplot as plt
import collections
import data.iris as dataSource
from ch02_1_perceptron import Perceptron

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

def test_preceptron_convergence():
    '''
    Plots graph of convergence,
    how many epochs it takes to converge

    Result:
    ---------
    Converges after 6 epochs!
    '''
    data = get_data()
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(data.X, data.y)
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_,
             marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of missclassifications')
    plt.show()

