# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the digits dataset, which contains images of handwritten digits (0-9)
digits = load_digits()

# Display the shape of the data (number of samples and features)
print(digits.data.shape)  # (1797, 64), 1797 samples with 64 features (8x8 images)
print(digits.data[0])     # Print the pixel values of the first digit

# Reshape and plot the 9th digit as an image
plt.gray()  # Set the color map to gray
plt.matshow(digits.data[9].reshape(8, 8))  # Reshape the 9th digit and display it
plt.show()  # Show the plot

# Display unique target values (digits 0-9)
print(np.unique(digits.target))  # Unique labels for digits
print(digits.target[9])  # Print the target label for the 9th digit

# Create a DataFrame from the digit data for easier manipulation
df = pd.DataFrame(digits.data, columns=digits.feature_names)
X = df  # Features (pixel values)
y = digits.target  # Target labels (actual digits)

# Scale the features to have mean=0 and variance=1 for better model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fit and transform the data
print(X_scaled)  # Display the scaled data

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=30)

# Create and train a logistic regression model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)  # Fit the model on the training data
print(model.score(X_test, y_test))  # Evaluate the model on the test data

# Apply PCA (Principal Component Analysis) to reduce dimensionality while retaining 95% of variance
from sklearn.decomposition import PCA
pca = PCA(0.95)  # Retain 95% of the variance
X_pca = pca.fit_transform(X)  # Fit and transform the entire dataset
print(X_pca.shape)  # Display the new shape after PCA

# Display the explained variance ratio and the number of components used
print(pca.explained_variance_ratio_)  # Variance explained by each principal component
print(pca.n_components_)  # Total number of components retained

# Split the PCA-transformed data into training and testing sets
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=30)

# Create and train a logistic regression model on the PCA-transformed data
model = LogisticRegression(max_iter=1000)  # Increase max_iter to ensure convergence
model.fit(X_train_pca, y_train)  # Fit the model on PCA-transformed training data
print(model.score(X_test_pca, y_test))  # Evaluate the model on PCA-transformed test data

# Apply PCA again but this time reducing to 2 dimensions for visualization
pca = PCA(n_components=2)  # Reduce to 2 components
X_pca = pca.fit_transform(X)  # Fit and transform the original data
print(X_pca.shape)  # Display the new shape (number of samples, 2)
