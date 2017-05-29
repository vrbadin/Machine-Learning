import pandas as pd

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

