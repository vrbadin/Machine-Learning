from data.housing import get_data
import matplotlib.pyplot as plt
import numpy as np
from chapter_10.lr_ransac import lin_regplot

df = get_data()
X = df[['LSTAT']].values
y = df[['MEDV']].values

# 1. Perform decision tree regression and plot
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X, y)

sort_idx = X.flatten().argsort()
lin_regplot(X[sort_idx], y[sort_idx], tree)
plt.xlabel('% lower status of the population [LSTAT]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.show()
