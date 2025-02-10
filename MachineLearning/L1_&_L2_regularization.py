import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:\\Users\\ayush\\Downloads\\Melbourne_housing_FULL.csv")

colstouse = ['Suburb','Rooms','Type','Method','SellerG','Regionname','Propertycount',
             'Distance','CouncilArea','Bedroom2','Bathroom','Car','Landsize','BuildingArea','Price']

df = df[colstouse]

colstofillzero = ['Propertycount','Distance','Bedroom2','Bathroom','Car']
df[colstofillzero] = df[colstofillzero].fillna(0)

df.isnull().sum()

df['Landsize'] = df['Landsize'].fillna(df.Landsize.mean())
df['BuildingArea'] = df['BuildingArea'].fillna(df.BuildingArea.mean())

df.dropna(inplace=True)

df = pd.get_dummies(df, drop_first=True)

X = df.drop('Price', axis=1)
y = df['Price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train, y_train)

# now lets check if it is overfitting or underfitting
reg.score(X_train, y_train)  # 68%
reg.score(X_test, y_test)   # 13%
# it is clearly overfitting

from sklearn import linear_model
lassoreg = linear_model.Lasso(alpha=50, max_iter=100, tol=0.1)
lassoreg.fit(X_train, y_train)

lassoreg.score(X_train, y_train)
lassoreg.score(X_test, y_test)

ridgereg = linear_model.Ridge(alpha=50, max_iter=100, tol=0.1)
ridgereg.fit(X_train, y_train)
ridgereg.score(X_train, y_train)
ridgereg.score(X_test, y_test)

