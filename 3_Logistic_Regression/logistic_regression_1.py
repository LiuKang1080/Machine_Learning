import numpy as np


N = 100
D = 2

X = np.random.randn(N, D)
# add a column of ones to the original data and include the bias term in the weight w
ones = np.array([[1] * N]).T
# Now concatenate ones and X
Xb = np.concatenate((ones, X), axis=1)

# initialize random weight vector:
w = np.random.randn(D + 1)

z = Xb.dot(w)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


print(sigmoid(z))
