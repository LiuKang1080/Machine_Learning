import pandas as pd


t1 = pd.read_csv('table1.csv')
t2 = pd.read_csv('table2.csv')

# To join in pandas, we use the merge() function
m = pd.merge(t1, t2, on='user_id')

print(m)
