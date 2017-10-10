import pandas as pd
from sklearn.preprocessing import LabelEncoder

def get_data():
    df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)
    return df_wine

def get_train_test():
    # 1. Get the breast cancer data
    df_bc = get_data()
    X, y = df_bc.iloc[:, 2:].values, df_bc.iloc[:, 1].values
    le = LabelEncoder()
    y = le.fit_transform(y)

    # 2. Split into training and testing data randomly
    #    30% test data / 70% training data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    return X_train, X_test, y_train, y_test