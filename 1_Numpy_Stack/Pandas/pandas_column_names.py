import pandas as pd
from datetime import datetime


df = pd.read_csv('international-airline-passengers.csv', engine='python', skipfooter=3)
df.columns = ['month', 'passengers']

# use the apply() function. convert the time on column 'month' into a new format with a new column
df['dt'] = df.apply(lambda row: datetime.strptime(row['month'], "%Y-%m"), axis=1)

print(df.info())
