import numpy as np

a = np.array([1, 2])
b = np.array([2, 1])

dot = 0
for i, j in zip(a, b):
    dot += i*j

print(dot)
# ==========================

# Now we will do this with the numpy array
print(np.sum(a*b))
print((a*b).sum())

# The best way:
print(np.dot(a, b))

# Find angle in between a and b
cos_angle = a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))
angle = np.arccos(cos_angle)
print(angle * 100)

# The default is in radians.
