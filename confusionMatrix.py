import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns

# Load the digits dataset
digits = load_digits()

# Check the attributes and methods of the digits object
dir(digits)

# Print the first data sample to see its structure
digits.data[0]

# Display the first 5 images from the dataset
plt.gray()  # Set the color map to grayscale
for i in range(5):
    plt.matshow(digits.images[i])  # Display each image
    plt.show(block=True)  # Show the image with blocking to prevent overlap

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)

# Print the number of samples in the training and test sets
len(X_train)  # Number of training samples
len(X_test)   # Number of test samples

# Create and train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)  # Fit the model with training data

# Evaluate the model on the test set
model.score(X_test, y_test)  # Print the accuracy score of the model

# Display a specific image and its predicted label
plt.matshow(digits.images[68])  # Display the image at index 68
plt.show(block=True)  # Show the image with blocking
digits.target[68]  # Print the actual label of the image

# Predict the label for the image at index 68 and the first 5 images
model.predict([digits.data[68]])  # Predict the label of the single image
model.predict(digits.data[0:5])  # Predict the labels for the first 5 images

y_predicted = model.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
cm

plt.figure(figsize= (10,7))
sns.heatmap(cm, annot=True)
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show(block=True)