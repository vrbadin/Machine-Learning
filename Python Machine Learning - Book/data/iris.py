import pandas as pd
from sklearn import datasets

def get():
    '''
    Iris dataset from UCI Machine Learning Repository
    '''
    df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                     'machine-learning-databases/iris/iris.data', header=None)
    return df

def test_loading():
    '''
    Return last 5 lines to check correct data loading
    '''
    return get().tail()

def get_skl():
    '''
    Get the data via scikit-learn
    '''
    iris = datasets.load_iris()
    return iris

