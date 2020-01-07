import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data_frame = pd.read_excel('mlr02.xls')
X = data_frame.as_matrix()

# See if there is a relationship between age and blood pressure.
plt.scatter(X[:, 1], X[:, 0])
plt.show()

# See if there is a relationship between weight and blood pressure.
plt.scatter(X[:, 2], X[:, 0])
plt.show()

# add our bias of ones
data_frame['ones'] = 1
Y = data_frame['X1']
X = data_frame[['X2', 'X3', 'ones']]

# compute 3 linear regressions
X2_only = data_frame[['X2', 'ones']]
X3_only = data_frame[['X3', 'ones']]


def get_R2(X, Y):
    w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
    Y_hat = np.dot(X, w)

    d1 = Y - Y_hat
    d2 = Y - Y.mean()

    R2 = 1 - d1.dot(d1) / d2.dot(d2)
    return R2


print("X2 only: ", get_R2(X2_only, Y))
print("X3 only: ", get_R2(X3_only, Y))
print("R2 for both: ", get_R2(X, Y))
