import matplotlib.pyplot as plt
from chapter_2.perceptron import Perceptron
from chapter_2 import helper_functions as fun


def get_perceptron():
    '''
    Initialises an instance of perceptron
    '''
    data = fun.get_data()
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(data.X, data.y)
    return ppn

def test_convergence():
    '''
    Plots graph of convergence,
    how many epochs it takes to converge

    Result:
    ---------
    Converges after 6 epochs!
    '''
    ppn = get_perceptron()
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_,
             marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of missclassifications')
    plt.show()

def plot_decision_regions():
    '''
    Plots decision boundaries
    '''
    data = fun.get_data()
    ppn  = get_perceptron()
    fun.plot_decision_regions(data.X, data.y, classifier=ppn)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()




