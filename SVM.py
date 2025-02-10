# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Loading the Iris dataset from sklearn
iris = load_iris()

# Displaying the list of attributes and methods of the iris object
# This can be useful to understand the structure of the dataset object
dir(iris)

# Accessing the feature names of the dataset
# This will provide a list of names for the features (columns) in the dataset
iris.feature_names

# Creating a DataFrame from the Iris dataset
# 'iris.data' contains the feature data, and 'iris.feature_names' provides column names
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Adding a new column to the DataFrame for the target values
# 'iris.target' contains the target labels (classes) for each sample
df['target'] = iris.target

# Accessing the names of the target classes
# This will provide the names of the classes for the target labels
iris.target_names

df['flower_names'] = df.target.apply(lambda x: iris.target_names[x])
df

from matplotlib import pyplot as plt
df0 = df[df.target==0]
df1 = df[df.target==1]
df2 = df[df.target==2]

plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'], color='green', marker="+")
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'], color='blue', marker="+")
plt.show(block=True)
plt.scatter(df0['petal length (cm)'],df0['petal width (cm)'], color='green', marker="+")
plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'], color='blue', marker="+")
plt.show(block=True)

from sklearn.model_selection import train_test_split
X= df.drop(['target', 'flower_names'], axis=1)
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


from sklearn.svm import SVC
model = SVC(C=20)
model.fit(X_train, y_train)
model.score(X_test, y_test)






# ABOUT THE CONCEPT OF SVM
#
# Support Vector Machines (SVM) are a type of machine learning model used to classify data into different categories.
# Think of it like drawing a line (or a more complex shape) that separates different types of data points.
#
# Key Points:
#
# 1. **Hyperplane**: This is the line (or shape) that SVM uses to separate different classes of data.
# In 2D, it's a line; in 3D, it's a plane; and in higher dimensions, it’s a hyperplane.
#
# 2. **Margin**: The margin is the distance between this separating line and the closest data
# points from each class. SVM tries to make this margin as wide as possible to ensure better separation.
#
# 3. **Support Vectors**: These are the closest data points to the line or shape. They are the
# most important points because they define where the line or shape is placed.
#
# 4. **Kernel Trick**: Sometimes, data can't be separated easily with a straight line. SVM uses
# something called kernels to transform the data into a higher dimension where a straight line can
# do the job. This makes it possible to handle more complex data.
#
# 5. **Regularization Parameter (C)**: This parameter controls how strict SVM is about fitting
# the training data. A high value of C means SVM will try to correctly classify all training
# examples, which might lead to overfitting. A low value of C allows some errors, which might lead
# to better generalization on new data.
#
# SVMs are powerful tools for classification because they aim to find the best possible separation
# between different classes of data.
