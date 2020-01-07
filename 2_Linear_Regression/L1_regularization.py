import numpy as np
import matplotlib.pyplot as plt


N = 50
D = 50

X = (np.random.randn(N, D) - 0.5) * 10
true_w = np.array([1, 0.5, -0.5] + [0] * (D-3))
Y = X.dot(true_w) + np.random.randn(N) * 0.5

costs = []
w = np.random.randn(D) / np.sqrt(D)
learning_rate = 0.001
L1 = 10.0

for t in range(500):
    Y_hat = X.dot(w)
    delta = Y_hat - Y
    w = w - learning_rate * (X.T.dot(delta) + L1 * np.sign(w))
    mse = delta.dot(delta) / N
    costs.append(mse)

plt.plot(costs)
plt.show()

plt.plot(true_w, label='true_w')
plt.plot(w, label='w_map')
plt.legend()
plt.show()
