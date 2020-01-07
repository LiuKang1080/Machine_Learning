import numpy as np


np.set_printoptions(suppress=True)

# Inverse of matrix A:
a = np.array(([1, 2], [3, 4]))
a_inv = np.linalg.inv(a)
# print(a_inv)
# print(a_inv.dot(A))

# Matrix determinants:
print(np.linalg.det(a))

# matrix Trace - sum of the diagonals of the matrix.
print(np.diag(a).sum())
print(np.trace(a))

# Finding the covariance of data set x
x = np.random.randn(100, 3)
# cov = np.cov(x)
# print(cov.shape)

# Here the cov shape is 100 x 100, but this is wrong, it should be 3 x 3 since our data set has 3 dimensions.
cov = np.cov(x.T)
# print(cov)
# Now this is a 3 x 3 matrix.

# there are 2 ways to find the eigenvalues and eigenvectors (eig, and eigh)
# Covariant matrices are symmetric, so we can use eigh. eigh is only used on Symmetric and Hermitian matrices.
print(np.linalg.eigh(cov))
print(np.linalg.eig(cov))
