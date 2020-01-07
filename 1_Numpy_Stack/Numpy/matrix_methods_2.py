import numpy as np


np.set_printoptions(suppress=True)

A = np.array([[1, 2],
              [3, 4]])

b = np.array([1, 2])

# Solution to solving systems of linear equations
x = np.linalg.inv(A).dot(b)
# print(x)

# Naturally Numpy has this built in.
y = np.linalg.solve(A, b)
# print(y)
# We get the same thing!

# =========================================================

# Admission at a fair is $1.50 for children and $4.00 for adults. On a certain day we have 2200 people, and $5,050
# was collected. How many children and how may adults?

A = np.array([[1, 1],
              [1.5, 4]])
b = np.array([2200, 5050])
print(np.linalg.solve(A, b))
