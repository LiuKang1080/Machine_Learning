import numpy as np
from Logistic_Regression_in_Python.ecommerce_project_process import get_binary_data


X, Y = get_binary_data()

D = X.shape[1]
w = np.random.randn(D)
# bias term:
b = 0


# create our sigmoid function:
def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def forward(X, w, b):
    return sigmoid(X.dot(w) + b)


P_Y_given_X = forward(X, w, b)
predictions = np.round(P_Y_given_X)


# Make function for our classification rate:
def classification_rate(Y, P):
    return np.mean(Y == P)


print("Score: ", classification_rate(Y, predictions))
