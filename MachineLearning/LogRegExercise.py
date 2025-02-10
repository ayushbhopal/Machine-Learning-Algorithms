import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\ayush\\OneDrive\\Desktop\\files\\Iris.csv")
df
df.isnull().sum()

X = df[['SepalLengthCm', 'SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y = df.Species

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train
y_train

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
model.score(X_test, y_test)
model.predict([[4.4, 2.9,1.4,0.2]])