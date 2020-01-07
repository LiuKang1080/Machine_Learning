import numpy as np
import matplotlib.pyplot as plt


X = []
Y = []

for line in open('data_1d.csv'):
    x, y = line.split(',')
    X.append(float(x))
    Y.append(float(y))

X = np.array(X)
Y = np.array(Y)

denom = X.dot(X) - X.mean() * X.sum()
a = (X.dot(Y) - Y.mean() * X.sum()) / denom
b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y)) / denom
Y_hat = a * X + b

# Calculation R^2:
d1 = Y - Y_hat
d2 = Y - Y.mean()

R2 = 1 - (d1.dot(d1) / d2.dot(d2))
print(R2)
