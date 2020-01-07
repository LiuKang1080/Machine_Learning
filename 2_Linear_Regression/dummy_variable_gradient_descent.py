# Dummy variable trap, and how to avoid this by using Gradient Descent
import numpy as np
import matplotlib.pyplot as plt


N = 10
D = 3

X = np.zeros((N, D))
X[:, 0] = 1
X[:5, 1] = 1
X[5:, 2] = 1

# Y is our targets
Y = np.array([0] * 5 + [1] * 5)

# try to solve for w:
try:
    weights = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
except Exception as err:
    print("could not solve " + str(err))

# as expected since it's a singular matrix we get a LinAlgError.
# we solve with gradient descent instead:

costs = []
w = np.random.randn(D) / np.sqrt(D)
learning_rate = 0.001

Y_hat = None

for t in range(1000):
    Y_hat = X.dot(w)
    delta = Y_hat - Y
    w = w - learning_rate * X.T.dot(delta)
    mean_squared_error = delta.dot(delta) / N
    costs.append(mean_squared_error)

plt.plot(costs)
plt.show()

# print final w. this is the solution to this problem
print(w)

# also you can confirm the solution by plotting Y and Y_hat:
plt.plot(Y_hat, label='Prediction')
plt.plot(Y, label='Targets')
plt.legend()
plt.show()
