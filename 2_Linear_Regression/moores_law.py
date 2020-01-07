import re
import numpy as np
import matplotlib.pyplot as plt


X = []
Y = []

non_decimal = re.compile(r'[^\d]+')

for line in open('moore.csv'):
    r = line.split('\t')

    x = int(non_decimal.sub('', r[2].split('[')[0]))
    y = int(non_decimal.sub('', r[2].split('[')[0]))
    X.append(x)
    Y.append(y)

X = np.array(X)
Y = np.array(Y)

Y = np.log(Y)
# plt.scatter(X, Y)
# plt.show()

denom = X.dot(X) - X.mean() * X.sum()
a = (X.dot(Y) - Y.mean() * X.sum()) / denom
b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y)) / denom
Y_hat = a * X + b

plt.scatter(X, Y)
plt.plot(X, Y_hat)
plt.show()

# Determine how good the model is: Compute the R^2
d1 = Y - Y_hat
d2 = Y - Y.mean()
R2 = 1 - (d1.dot(d1) / d2.dot(d2))
print(R2)

# How long will it take for transistor count to double?

# log(tc)      = a * year + b
# tc           = exp(b) * exp(a * year)
# 2*tc         = 2 * exp(b) * exp(a * year)    = exp(ln(2) * exp(b) * exp(a * year)
#              = exp(b) * exp(a * year) + ln(2)
# exp(b) * exp(a * year_2)  = exp(b) * exp(a * year_1 + ln(2))
# year_2 = year_1 + ln(2) / a
# np.log(2) / a

print("Time for transistor count to double: ")
print(np.log(2) / a)
