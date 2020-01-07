import numpy as np


data_list = []

for line in open('data_2d.csv'):
    data_line = line.rstrip()           # Gets rid of the '\n' at the end of each line
    row = data_line.split(',')          # Results in a list with 3 elements

    data_list.append(row)

A = np.array(data_list)

