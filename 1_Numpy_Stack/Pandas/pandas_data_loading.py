import pandas as pd


X = pd.read_csv('data_2d.csv', header=None)
# print(X.info())

# show the first few rows.
# print(X.head())
# if we want to specify how many rows, pass in number as argument.
# print(X.head(10))

# Convert Data set into NumPy array.
M = X.as_matrix()
# print(type(M))

# print(type(X[0]))

# Get the 0th row. 2 ways
# print(X.iloc[0])
# print(X.ix[0])

# Select specific columns. 0th and 2nd row
# print(X[[0, 2]])
