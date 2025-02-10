
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\ayush\\Downloads\\salaries.csv")
df

inputs = df.drop('salary_more_then_100k', axis=1)
target = df['salary_more_then_100k']

from sklearn.preprocessing import LabelEncoder
le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()


inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['jon_n'] = le_company.fit_transform(inputs['job'])
inputs['degree_n'] = le_company.fit_transform(inputs['degree'])
inputs.head()

inputs_n = inputs.drop(['company', 'job', 'degree'], axis=1)
inputs_n

from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(inputs_n, target)

model.score(inputs_n,target)
model.predict([[2,0,0]])





# ABOUT THE CONCEPT OF DECISION TREE CLASSIFICATION
#
# **Decision Tree Classifier** is a supervised learning algorithm used for classification tasks.
# It builds a tree-like model of decisions based on the features of the dataset.
#
# Key Concepts:
#
# 1. **Decision Tree**: A decision tree splits the data into branches based on feature
# values. Each node represents a feature (or attribute), each branch represents a decision
# rule, and each leaf node represents an outcome (class label).
#
# 2. **Feature Encoding**:
#    - **Label Encoding**: This converts categorical features into numerical values.
#    For example, 'company', 'job', and 'degree' are converted into numerical codes that
#    the model can use.
#
# 3. **Training**: The `fit` method trains the decision tree model using the features
# (inputs) and the target labels (outcomes). The tree learns how to split the data to b
# est predict the target.
#
# 4. **Prediction**: Once the model is trained, it can make predictions on new data.
# For example, `model.predict([[2,0,0]])` uses the trained model to predict the target
# class for the input `[2, 0, 0]`.
#
# 5. **Model Evaluation**: The `score` method evaluates the model's accuracy on the
# training data. It calculates the proportion of correctly classified samples.
#
# In this code, the decision tree is trained on encoded feature data to predict whether
# a salary is more than $100k based on job company, job role, and degree.
