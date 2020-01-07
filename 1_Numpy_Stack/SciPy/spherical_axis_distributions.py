import numpy as np
import matplotlib.pyplot as plt


# Sample from a 2-D gaussian with mean = 0, variance = 1 (Spherical Gaussian).
r = np.random.randn(10000, 2)
# plt.scatter(r[:, 0], r[:, 1])
# plt.axis('equal')
# plt.show()

# Elliptical Gaussian distribution, Standard deviation = 5, and mean = 2 for column 2.
r[:, 1] = 5 * r[:, 1] + 2
plt.scatter(r[:, 0], r[:, 1])
plt.axis('equal')
plt.show()

# As we can see the regular distribution is circular, and elliptical one is eggy.
