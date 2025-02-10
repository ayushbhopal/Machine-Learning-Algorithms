# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model

# Load data from a CSV file into a DataFrame
df = pd.read_csv("C:\\Users\\ayush\\Downloads\\homeprices.csv")

# Display the first few rows of the DataFrame for inspection (optional, can be commented out)
print(df.head())

# Fill missing values in the 'bedrooms' column with the median of the column
df.bedrooms = df.bedrooms.fillna(df.bedrooms.median())

# Initialize and train a linear regression model
reg = linear_model.LinearRegression()
reg.fit(df[['area', 'bedrooms', 'age']], df.price)

# Output the coefficients and intercept of the trained model
print(f"Coefficients: {reg.coef_}")
print(f"Intercept: {reg.intercept_}")

# Use the trained model to predict the price for a given set of features
predicted_price = reg.predict([[3000, 3, 40]])
print(f"Predicted price for area=3000, bedrooms=3, age=40: {predicted_price[0]}")

# Manually calculate the predicted price using the formula
manual_prediction = (reg.coef_[0] * 3000 +
                     reg.coef_[1] * 3 +
                     reg.coef_[2] * 40 +
                     reg.intercept_)
print(f"Manual calculation of predicted price: {manual_prediction}")

# Create a pair plot to visualize relationships between features
sns.pairplot(df)
plt.show(block=True)


import joblib
joblib.dump(reg, 'model_joblib')

mj = joblib.load('model_joblib')
mj.predict([[5000]])