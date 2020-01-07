import numpy as np
import matplotlib.pyplot as plt


# Sample from a normal gaussian distribution
r = np.random.randn(10000)
plt.hist(r, bins=100)
# plt.show()

# Sample from a normal distribution with arbitrary mean and standard deviation.
R = 10 * np.random.randn(10000) + 5
# Standard Deviation = 10
# Mean = 5
plt.hist(R, bins=100)
plt.show()
