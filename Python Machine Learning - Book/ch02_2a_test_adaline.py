import numpy as np
import matplotlib.pyplot as plt
from ch02_2_adaline import AdalineGD
import ch02_helper_functions as fun

def test_convergence():
    '''
    Plots graphs of convergence for 2 etas:
    0.01 and 0.0001.

    Result:
    --------
    2 issues:
    Case eta=0.01 gets worse with epochs
    as steepest descent overshoots due to large steps.
    Case eta=0.0001 converges with increase of epochs,
    but the steps are too small and require lots of steps.
    '''
    data = fun.get_data()
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    adal_0 = AdalineGD(n_iter=10, eta=0.01).fit(data.X, data.y)
    ax[0].plot(range(1, len(adal_0.cost_) + 1),
               np.log10(adal_0.cost_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Sum-squared-error)')
    ax[0].set_title('Adaline - Learning rate 0.01')

    adal_1 = AdalineGD(n_iter=10, eta=0.0001).fit(data.X, data.y)
    ax[1].plot(range(1, len(adal_1.cost_) + 1),
               np.log10(adal_1.cost_), marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('log(Sum-squared-error)')
    ax[1].set_title('Adaline - Learning rate 0.0001')
    plt.show()

def test_convergence_standardised():
    '''
    Plots graph of convergence for
    standardised X and eta=0.01.

    Result:
    --------
    Improved convergence!
    '''
    data = fun.get_data()
    X_std = fun.standardise(data.X)
    adal = AdalineGD(n_iter=15, eta=0.01)
    adal.fit(X_std, data.y)
    fun.plot_decision_regions(X_std, data.y, classifier=adal)
    plt.title('Adaline - Gradient Descent')
    plt.xlabel('sepal length [standardised]')
    plt.ylabel('petal length [standardised]')
    plt.legend(loc='upper left')
    plt.show()
    plt.plot(range(1, len(adal.cost_) + 1),
               np.log10(adal.cost_), marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('log(Sum-squared-error)')
    plt.show()
