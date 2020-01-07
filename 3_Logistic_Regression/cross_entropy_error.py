import numpy as np
import matplotlib.pyplot as plt


N = 100
D = 2

X = np.random.randn(N, D)

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


print(cross_entropy_error(T, Y))

# Use the closed form solution yo logistic regression and see how good that solution is.
# This will work because we have equal variances in both classes. Variance = 1, which is the default.
w = np.array([0, 4, 4])
z = Xb.dot(w)
Y = sigmoid(z)

print(cross_entropy_error(T, Y))

# now we will plot, we expect the line to take the form y=-x
plt.scatter(X[:, 0], X[:, 1], c=T, s=100, alpha=0.5)
# draw the line:
x_axis = np.linspace(-6, 6, 100)
y_axis = -x_axis
plt.plot(x_axis, y_axis)
plt.show()
