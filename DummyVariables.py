# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model

df = pd.read_csv("C:\\Users\\ayush\\Downloads\\homeprices (1).csv")
df
dummies = pd.get_dummies(df.town).astype(int)
dummies

merged = pd.concat([df, dummies], axis='columns')
merged

final = merged.drop(['town', 'west windsor'], axis='columns')
final

from sklearn.linear_model import LinearRegression
model = LinearRegression()

X = final.drop('price', axis='columns')

y = final.price

model.fit(X,y) # training your model
model.predict([[2800, 0, 1]])
model.score(X,y)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dfle = df
dfle.town = le.fit_transform(dfle.town)
dfle

X = dfle[['town', 'area']].values  # we want it to be a 2D array (not a dataframe)
y = dfle.price

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
X_encoded = ohe.fit_transform(X).toarray()
X_encoded
