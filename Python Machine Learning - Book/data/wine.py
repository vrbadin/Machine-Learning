import pandas as pd

def get_data():
    df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
    return df_wine

def get_train_test():
    # 1. Get the wine data
    df_wine = get_data()
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

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

    return X_train_std, X_test_std, y_train, y_test