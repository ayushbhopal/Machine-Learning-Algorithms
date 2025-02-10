# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import pickle

# Load data from an Excel file into a DataFrame
df = pd.read_excel("C:\\Users\\ayush\\OneDrive\\Desktop\\files\\LR.xlsx")

# Display the DataFrame (for inspection, not typically used in scripts)
print(df)

# Create a scatter plot to visualize the relationship between 'area' and 'price'
plt.scatter(df.area, df.price, color='red', marker="+")
plt.xlabel('Area')  # Label for the x-axis
plt.ylabel('Price')  # Label for the y-axis
plt.title('Scatter Plot of Area vs Price')  # Title of the plot
plt.show(block=True)  # Display the plot

# Initialize and train a linear regression model
reg = linear_model.LinearRegression()
reg.fit(df[['area']], df['price'])  # Fit the model with 'area' as the feature and 'price' as the target

# Make a prediction for an area of 3300
predicted_price = reg.predict([[3300]])
print(f"Predicted price for an area of 3300: {predicted_price[0]}")

# Display the coefficients and intercept of the fitted model
print(f"Coefficient (slope): {reg.coef_[0]}")  # Coefficient (slope) of the linear regression line
print(f"Intercept: {reg.intercept_}")  # Intercept of the linear regression line

# Manually calculate the predicted price using the formula y = mx + b
# where m is the coefficient and b is the intercept
manual_prediction = reg.coef_[0] * 3300 + reg.intercept_
print(f"Manual calculation of predicted price: {manual_prediction}")

# Load new data from another Excel file
d = pd.read_excel("C:\\Users\\ayush\\OneDrive\\Desktop\\files\\LR2.xlsx")

# Predict prices for the new data using the trained model
p = reg.predict(d[['area']])  # Ensure 'area' column is used for prediction

# Add the predicted prices to the new DataFrame
d['Price'] = p

# Save the updated DataFrame with predicted prices to a new Excel file
d.to_excel("C:\\Users\\ayush\\OneDrive\\Desktop\\files\\LR3.xlsx", index=False)

# Plot the original data and the linear regression line
plt.xlabel('Area')
plt.ylabel('Price')
plt.scatter(df.area, df.price, color='red', marker="+")  # Plot original data points
plt.plot(df.area, reg.predict(df[['area']]), color='blue')  # Plot the linear regression line
plt.show(block=True)

# Save the trained model using pickle
with open('model_pickle.pkl', 'wb') as f:
    pickle.dump(reg, f)  # Save the model to a file

# Load the trained model using pickle
with open('model_pickle.pkl', 'rb') as f:
    mp = pickle.load(f)  # Load the model from the file

# Predict using the loaded model
print(mp.predict([[5000]]))  # Predict the price for an area of 5000 using the loaded model


