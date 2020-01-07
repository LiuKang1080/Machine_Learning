import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


A = pd.read_csv('data_1d.csv', header=None).as_matrix()

# 1st column is the x-axis, 2nds column is the y-axis.
x = A[:, 0]
y = A[:, 1]

plt.scatter(x, y)

x_line = np.linspace(0, 100, 100)
y_line = 2 * x_line + 1

plt.plot(x_line, y_line)

plt.show()
