import numpy as np


N = 100
D = 2       # dimensionality = 2

X = np.random.randn(N, D)   # create a random gaussian distribution

X[:50, :] = X[:50, :] - 2 * np.ones((50, D))
X[50:, :] = X[50:, :] + 2 * np.ones((50, D))

# array of targets, first 50 = 0, next 50 = 1
T = np.array([0]*50 + [1]*50)

# column of ones
ones = np.array([[1]*N]).T
Xb = np.concatenate((ones, X), axis=1)

# randomly initialize weights
w = np.random.randn(D+1)

z = Xb.dot(w)


# sigmoid function:
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


Y = sigmoid(z)


# write the cross entropy error function:
def cross_entropy_error(T, Y):
    """
    Calculates the error using the cross entropy error model.
    :param T: Targets
    :param Y: Predicted Output
    :return: E - Error
    """
    E = 0
    for i in range(N):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])

    return E


learning_rate = 0.1
for i in range(100):
    # print the cross entropy error every 10 steps to see that it's decreasing
    if i % 10 == 0:
        print(cross_entropy_error(T, Y))

    w += learning_rate * np.dot((T - Y).T, Xb)
    Y = sigmoid(Xb.dot(w))

print("Final weight is:", w)
