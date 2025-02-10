# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from an Excel file
df = pd.read_excel("C:\\Users\\ayush\\OneDrive\\Desktop\\files\\BMW.xlsx")

# Select features and target variable
X = df[['Mileage', 'Age(yrs)']]  # Features
y = df[['Sell Price($)']]         # Target variable

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)  # Set random_state for reproducibility

# Display the number of samples in training and testing sets
len(X_train)  # Number of training samples
len(X_test)   # Number of testing samples

# Initialize and fit the Linear Regression model
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)  # Train the model

# Make predictions on the test set
predictions = reg.predict(X_test)

# Display the true values of the test set
y_test

# Evaluate the model performance using R-squared score
score = reg.score(X_test, y_test)
score  # R-squared score of the model

