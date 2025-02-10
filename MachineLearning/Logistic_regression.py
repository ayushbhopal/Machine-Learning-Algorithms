# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import pickle

# Load data from a CSV file into a DataFrame
df = pd.read_csv("C:\\Users\\ayush\\Downloads\\insurance_data.csv")

# Display the first few rows of the DataFrame
df

# Plot a scatter plot of age vs. bought_insurance
plt.scatter(df.age, df.bought_insurance, marker='+', color='red')
plt.xlabel('Age')
plt.ylabel('Bought Insurance')
plt.title('Age vs Bought Insurance')
plt.show(block=True)  # Display the plot

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[['age']], df.bought_insurance, train_size=0.9, random_state=10)  # Set random_state for reproducibility

# Display the training and testing data
X_test  # Testing features
X_train  # Training features

# Initialize and train the Logistic Regression model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)  # Train the model

# Make predictions on the test set
predictions = model.predict(X_test)
predictions
# Evaluate the model's accuracy on the test set
accuracy = model.score(X_test, y_test)
accuracy  # Accuracy score of the model

# Predict the probability of customers buying insurance on the test set
probabilities = model.predict_proba(X_test)
probabilities  # Probability estimates for the test data
