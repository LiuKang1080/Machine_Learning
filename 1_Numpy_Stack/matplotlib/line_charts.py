import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(0, 10, 100)
y = np.sin(x)

# now we plot the sin wave
plt.plot(x, y)

# We can add things to the plot before showing.
plt.xlabel("Time")
plt.ylabel("Magnitude")
plt.title("Data Set 1")

plt.show()
