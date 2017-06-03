import matplotlib.pyplot as plt
from chapter_2.adaline_sgd import AdalineSGD
from chapter_2 import helper_functions as fun


def test_convergence():
    '''
    Plots graphs of convergence.

    Result:
    --------
    After 15 epochs, we get the similar results as we do
    with the bach Gradient Descent.
    '''
    data = fun.get_data()
    X_std = fun.standardise(data.X)

    adal = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
    adal.fit(X_std, data.y)
    fun.plot_decision_regions(X_std, data.y, classifier=adal)
    plt.title('Adaline - Stochastic Gradient Descent')
    plt.xlabel('sepal length [standardised]')
    plt.ylabel('petal length [standardised]')
    plt.legend(loc='upper left')
    plt.show()
    plt.plot(range(1, len(adal.cost_) + 1),
             adal.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Average Cost')
    plt.show()

test_convergence()