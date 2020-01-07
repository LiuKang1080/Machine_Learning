import numpy as np


np.set_printoptions(suppress=True)

A = np.array([[1, 2], [4, 3]])
print(np.linalg.eig(A))
