import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the diabetes dataset from a CSV file
df = pd.read_csv("C:\\Users\\ayush\\Downloads\\diabetes.csv")

# Check for any missing values in the dataset
df.isnull().sum()

# Get a statistical summary (count, mean, std, min, max, etc.) of the dataset's features
df.describe()

# Count the number of occurrences of each class (0 or 1) in the 'Outcome' column
df.Outcome.value_counts()

# Separate the features (X) from the target variable (y)
X = df.drop('Outcome', axis=1)  # All columns except 'Outcome'
y = df.Outcome                   # The target variable

# Standardize the feature values to have a mean of 0 and a standard deviation of 1
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fit and transform the features

# Split the dataset into training and testing sets, while maintaining the same class distribution
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, random_state=10)

# Import DecisionTreeClassifier and cross_val_score for model evaluation
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation on a Decision Tree classifier to assess its performance
scores = cross_val_score(DecisionTreeClassifier(), X, y, cv=5)
# Calculate the mean score from the cross-validation results
scores.mean()

# Import BaggingClassifier for ensemble learning to improve model performance
from sklearn.ensemble import BaggingClassifier

# Initialize BaggingClassifier with DecisionTreeClassifier as the base estimator
bag_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(),  # Use a Decision Tree as the base model
    n_estimators=100,                    # Set the number of base models to train
    max_samples=0.8,                     # Specify the fraction of samples to use for each model
    oob_score=True,                      # Enable out-of-bag score estimation for better validation
    random_state=0                       # Set random seed for reproducibility of results
)

# Fit the Bagging model to the training data
bag_model.fit(X_train, y_train)

# Retrieve the out-of-bag score, which provides an estimate of model performance on unseen data
bag_model.oob_score_

# Evaluate the accuracy of the Bagging model on the test dataset
bag_model.score(X_test, y_test)

# Perform 5-fold cross-validation on the Bagging model to assess its performance
scores = cross_val_score(bag_model, X, y, cv=5)
# Calculate the mean cross-validation score for the Bagging model
scores.mean()

# Import RandomForestClassifier for comparison with the Bagging model
from sklearn.ensemble import RandomForestClassifier

# Perform 5-fold cross-validation on a Random Forest classifier to assess its performance
scores = cross_val_score(RandomForestClassifier(), X, y, cv=5)
# Calculate the mean score from the cross-validation results for the Random Forest
scores.mean()
