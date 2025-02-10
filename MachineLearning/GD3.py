
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model

# Load data from a CSV file into a DataFrame
df = pd.read_csv("C:\\Users\\ayush\\Downloads\\test_scores.csv")


reg = linear_model.LinearRegression()
reg.fit(df[['math']], df['cs'])
reg.coef_    # 1.017736
reg.intercept_  # 1.91521


def gradientdescent(x,y):
    m_curr = b_curr = 0
    iterations = 1000
    n = len(x)
    learning_rate = 0.0001

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1 / n) * sum([val ** 2 for val in (y - y_predicted)])
        md = -(2 / n) * sum(x * (y - y_predicted))
        bd = -(2 / n) * sum(y - y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd

        print('m {}, b {}, cost {}, iterations {}'.format(m_curr,b_curr,cost, i))


x = np.array(df.math)
y = np.array(df.cs)
gradientdescent(x,y)
