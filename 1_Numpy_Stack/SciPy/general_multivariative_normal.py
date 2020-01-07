import numpy as np
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt


cov = np.array([[1, 0.8], [0.8, 1]])
# Variance = 1 for the 1st dimension, and 3 for the 2nd. covariance = 0.8 between both dimensions.

# Crate a mean of 2
mu = np.array([0, 2])

r = mvn.rvs(mean=mu, cov=cov, size=1000)

plt.scatter(r[:, 0], r[:, 1])
plt.axis('equal')
plt.show()

# NumPy can also draw this:
r = np.random.multivariate_normal(mean=mu, cov=cov, size=1000)
plt.scatter(r[:, 0], r[:, 1])
plt.axis('equal')
plt.show()
