import numpy as np
import matplotlib.pyplot as plt


N = 1000
D = 2

# we have 2 radius inner and outer radius for the donut
r_inner = 5
r_outer = 10

# create a uniform distribution and store it into a variable.
# randomly creates a uniform distribution around a fixed point offsetted by an initial value of the radius
R1 = np.random.randn(int(N / 2)) + r_inner
# this generates the data around "5"
# generate some angles which will be in polar coordinates
# and they will be uniformly distributed
theta = 2 * np.pi * np.random.random(int(N / 2))

# convert polar coordinates into (x,y) coordinates
X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T
# notice the transpose at the end

# do the same thing for the outer radius
R2 = np.random.randn(int(N / 2)) + r_outer

# initialize theta again to be used for polar coordinates on the outer radius
theta = 2 * np.pi * np.random.random(int(N / 2))

# calculate x_outer the same way
X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

# calculate the total X
X = np.concatenate([X_inner, X_outer])
# initialize our targets, initial set of 0, and then a set of 1
T = np.array([0] * (int(N/2)) + [1] * (int(N/2)))

# now we plot the data to see what the donut problem looks like.
plt.scatter(X[:, 0], X[:, 1], c=T)
plt.show()

# Problem: We can't create a line that separates the 2 distinct regions of the donut.
# Though in fact we can create a solution to this for logistic regression

# create column of ones, then transpose it:
ones = np.array([[1] * N]).T

# the trick to the donut problem is that we will create another column
# this second column represents the radius of the point, this makes the data points linearly separable
# then we will manually count the radius of each data point
r = np.zeros((N, 1))

for i in range(N):
    r[i] = np.sqrt(X[i, :].dot(X[i, :]))

# now we concatenate the ones and the radii
Xb = np.concatenate((ones, r, X), axis=1)

# randomly initialize the weights
w = np.random.rand(D+2)

# the rest is the regular sigmoid and cross entropy functions
z = Xb.dot(w)


# sigmoid function:
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


Y = sigmoid(z)


# write the cross entropy error function:
def cross_entropy_error(T, Y):
    """
    Calculates the error using the cross entropy error model.
    :param T: Targets
    :param Y: Predicted Output
    :return: E - Error
    """
    E = 0
    for i in range(N):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])

    return E


# this learning rate was pre-determined by the LazyProgrammer
# the number of iterations was also done by the LazyProgrammer
# generally to get these numbers we experiment or use cross validation
learning_rate = 0.0001
error = []

for i in range(5000):
    e = cross_entropy_error(T, Y)
    error.append(e)
    if i % 100 == 0:
        print(e)

    # gradient descent with regularization
    w += learning_rate * (np.dot((T-Y).T, Xb) - 0.01*w)
    Y = sigmoid((Xb.dot(w)))

# plot the error as it evolves with time.
plt.plot(error)
plt.title("Cross-Entropy")
plt.show()

print("Final Weight: ", w)
print("Final Classification Rate: ", 1 - np.abs(T - np.round(Y)).sum() / N)

# some notes: look at our final weights they are both pretty close to 0 for both x and y
# this means that our model does not depend on the X and Y coordinates, it depends more on the bias
# (small radius ==> negative bias and pushes the classification towards zero)
# (large radius ==> pushes the classification towards one
