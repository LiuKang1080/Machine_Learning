import numpy as np

L = [1, 2, 3]
A = np.array([1, 2, 3])

for i in L:
    print(i)

for j in A:
    print(j)

B = A + A
print(B)

print(A**2)
print(np.sqrt(A))
print(np.log(A))
print(np.exp(A))
