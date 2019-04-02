import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from finaldatatransforms import data_cleanup

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
df = pd.read_csv('data/churn.csv')

df = data_cleanup(df)

X = df.iloc[:,:12]
y = df.iloc[:,12]

churn = df.groupby("phone")
print(churn.mean())