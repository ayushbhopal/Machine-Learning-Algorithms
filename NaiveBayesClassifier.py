import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\ayush\\Downloads\\titanic.csv")
df.drop(['PassengerId', 'Name','SibSp','Parch','Ticket','Cabin','Embarked'], axis=1, inplace=True)

target = df.Survived
inputs = df.drop('Survived', axis=1)
dummies = pd.get_dummies(inputs.Sex).astype(int)
inputs = pd.concat([inputs, dummies], axis='columns')
inputs.drop('Sex', axis=1, inplace=True)

inputs.columns[inputs.isna().any()]
inputs.Age[:10]

inputs.Age = inputs.Age.fillna(inputs.Age.mean())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)

model.score(X_test, y_test)
model.predict(X_test[:10])





# ABOUT THE CONCEPT OF NAIVE BAYES
#
# Naive Bayes is a simple and effective classification algorithm based on Bayes' Theorem.
# It assumes that the features used for classification are independent of each other, which is why it's called "naive."
#
# Key Concepts:
#
# 1. **Bayes' Theorem**: This is the foundation of Naive Bayes. It calculates the
# probability of a class given the features, based on prior knowledge of the class
# probabilities and the likelihood of the features given the class.
#
# 2. **Independence Assumption**: Naive Bayes assumes that each feature contributes
# independently to the probability of a class. This simplifies the computation, even
# though real-world data might not always meet this assumption.
#
# 3. **Classification**: The model calculates the probability of each class for a
# given set of features and assigns the class with the highest probability to the data point.
#
# Naive Bayes is particularly useful for text classification and works well with
# large datasets and high-dimensional data.
