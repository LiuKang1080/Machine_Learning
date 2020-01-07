from scipy.stats import norm
import numpy as np

# PDF - Probability Density Function
print(norm.pdf(0))

r = np.random.randn(10)

print(norm.pdf(r))
print(norm.logpdf(r))

# Cumulative Distribution Function
print(norm.cdf(r))
print(norm.logcdf(r))
