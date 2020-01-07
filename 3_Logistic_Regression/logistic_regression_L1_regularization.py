import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# We create a fat matrix, where N < D:
N = 50
D = 50

# Uniformly distributed between -5 and +5
X = (np.random.random((N, D)) - 0.5) * 10

# we use the -0.5 to center our data at 0, remember that our distribution starts around 50.

# only the first 3 dimensions affect the output. The rest 47 do not (they are 0s).
true_w = np.array([1, 0.5, -0.5] + [0]*(D-3))

# generate Y, our targets, with random noise:
Y = np.round(sigmoid(X.dot(true_w) + np.random.randn(N)*0.5))

# Now we need to perform gradient descent:
costs = []
# randomly initialize the weights.
w = np.random.randn(D) / np.sqrt(D)
learning_rate = 0.001
# now set up a L1 penalty:
L1 = 10.0

# 5,000 epochs:
for t in range(5000):
    Y_hat = sigmoid(X.dot(w))
    delta = Y_hat - Y

    w = w - learning_rate*(X.T.dot(delta) + L1*np.sign(w))

    # find and store the costs (target times the prediction).
    cost = -(Y*np.log(Y_hat) + (1-Y)*np.log(1 - Y_hat)).mean() + L1*np.abs(w).mean()
    # store in the list of costs
    costs.append(cost)

# plot the costs:
plt.plot(costs)
plt.show()

# plot the true w vs the w we found so that we can compare them.
plt.plot(true_w, label="true w")
plt.plot(w, label="w map")
plt.legend()
plt.show()


# We can see that the cost converges pretty quickly. Our weights don't match the true weights exactly with an L1 = 2.0.
# what if we try an L1 penalty = 10.0?
# we see that the costs converge quickly again, but the weights are now much closer to 0. we may want to try a
# smaller regularization.

