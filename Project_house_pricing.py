# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import seaborn as sns  # Unused library, consider removing

# Load the dataset
df1 = pd.read_csv("C:\\Users\\ayush\\Downloads\\archive (5)\\Bengaluru_House_Data.csv")

# Display the shape and column names of the DataFrame
print(df1.shape)
print(df1.columns)

# Check unique values and counts in the 'area_type' column
print(df1['area_type'].unique())
print(df1['area_type'].value_counts())

# Drop unnecessary columns for analysis
df2 = df1.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')
print(df2.shape)  # Display shape after dropping columns

# Check for missing values and remove rows with missing values
print(df2.isnull().sum())
df3 = df2.dropna()
print(df3.isnull().sum())  # Confirm no missing values

# Check unique values in the 'size' column and extract BHK
df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
print(df3.bhk.unique())

# Check unique values in the 'total_sqft' column
print(df3['total_sqft'].unique())


# Function to check if a value can be converted to float
def is_float(x):
    try:
        float(x)
    except ValueError:
        return False
    return True


# Display rows where 'total_sqft' cannot be converted to float
print(df3[~df3['total_sqft'].apply(is_float)].head(10))


# Function to convert 'total_sqft' values to numeric
def convert_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:  # Average if value is a range
        return (float(tokens[0]) + float(tokens[1])) / 2
    try:
        return float(x)  # Otherwise, return float value
    except ValueError:
        return None


# Apply conversion function
df4 = df3.copy()
df4.total_sqft = df4.total_sqft.apply(convert_to_num)
df4 = df4[df4.total_sqft.notnull()]  # Remove rows with null total_sqft
print(df4.head(2))

# Calculate price per square foot
df5 = df4.copy()
df5['price_per_sqft'] = df5['price'] * 100000 / df5['total_sqft']
print(df5.head(10))

# Process locations
df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending=False)

# Group locations with fewer than 10 listings into 'other'
location_stats_less_than_10 = location_stats[location_stats <= 10]
df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)

# Remove properties with less than 300 sqft per bhk
df6 = df5[~(df5.total_sqft / df5.bhk < 300)]
print(df6.shape)

# Describe price per sqft statistics
print(df6.price_per_sqft.describe())


# Function to remove outliers based on price per square foot
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out


# Remove outliers
df7 = remove_pps_outliers(df6)
print(df7.shape)


# Function to plot scatter charts for specific locations
def plot_scatter_chart(df, location):
    bhk2 = df[(df.location == location) & (df.bhk == 2)]
    bhk3 = df[(df.location == location) & (df.bhk == 3)]
    plt.figure(figsize=(15, 10))
    plt.scatter(bhk2.total_sqft, bhk2.price, color='blue', label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft, bhk3.price, marker='+', color='green', label='3 BHK', s=50)
    plt.xlabel('Total Square Feet Area')
    plt.ylabel('Price')
    plt.title(location)
    plt.legend()
    plt.show(block=True)  # Show the plot


# Plot scatter chart for a specific location
plot_scatter_chart(df7, 'Hebbal')


# Function to remove BHK outliers based on price per square foot
def remove_bhk_outliers(df):
    exclude_indices = []

    for location, location_df in df.groupby('location'):
        bhk_stats = {}

        # Calculate statistics for each BHK category
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        # Remove outliers for each BHK category
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk - 1)
            if stats and stats['count'] > 5:
                exclude_indices.extend(bhk_df[bhk_df.price_per_sqft < (stats['mean'])].index.values)

    return df.drop(index=exclude_indices)


# Remove BHK outliers
df8 = remove_bhk_outliers(df7)
print(df8.shape)

# Re-plot the scatter chart for a specific location
plot_scatter_chart(df8, 'Uttarahalli')

# Plot histogram of price per square foot
plt.figure(figsize=(20, 10))
plt.hist(df8.price_per_sqft, rwidth=0.8)
plt.xlabel('Price Per Square Feet')
plt.ylabel('Count')
plt.show(block=True)  # Show the plot

# Plot histogram of the number of bathrooms
plt.hist(df8.bath, rwidth=0.8)
plt.xlabel('Number of Bathrooms')
plt.ylabel('Count')
plt.show(block=True)  # Show the plot

# Filter DataFrame to exclude properties with excessive bathrooms
df9 = df8[df8.bath < df8.bhk + 2]
print(df9.shape)

# Drop unnecessary columns for the final dataset
df10 = df9.drop(['size', 'price_per_sqft'], axis='columns')
print(df10.head(3))

# Create dummy variables for categorical 'location' feature
dummies = pd.get_dummies(df10.location).astype(int)

# Concatenate dummy variables to the DataFrame
df11 = pd.concat([df10, dummies.drop('other', axis=1)], axis=1)

# Drop the original 'location' column from the final DataFrame
df12 = df11.drop('location', axis=1)

# Prepare features (X) and target variable (y)
X = df12.drop('price', axis=1)  # Features
y = df12.price  # Target variable

# Importing the model selection module
from sklearn.model_selection import train_test_split

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor

# Splitting the dataset into training and testing sets
# `X` is the feature set, `y` is the target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Initialize a Linear Regression model
lr_clf = LinearRegression()

# Fit the model on the training data
lr_clf.fit(X_train, y_train)

# Evaluate the model's performance on the test set
test_score = lr_clf.score(X_test, y_test)

# Initialize ShuffleSplit for cross-validation
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

# Perform cross-validation and obtain scores for Linear Regression
cross_val_scores = cross_val_score(LinearRegression(), X, y, cv=cv)


# Define a function to find the best model among several algorithms
def find_best_model(X, y):
    # Dictionary to hold model configurations
    algos = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
                # Removed normalize as it's deprecated in newer versions
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 2],  # Regularization parameter for Lasso
                'selection': ['random', 'cyclic']  # Method to choose the features
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['mse', 'friedman_mse'],  # Measure of quality for a split
                'splitter': ['best', 'random']  # Strategy used to choose the split at each node
            }
        }
    }

    # List to store the scores for each model
    scores = []

    # Initialize ShuffleSplit for cross-validation
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

    # Loop through each algorithm and perform Grid Search for hyperparameter tuning
    for algo_name, config in algos.items():
        # Set up GridSearchCV with the model and parameters
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        # Fit the model to the data
        gs.fit(X, y)
        # Append the results to the scores list
        scores.append({
            'model': algo_name,  # Name of the algorithm
            'best_score': gs.best_score_,  # Best score from cross-validation
            'best_params': gs.best_params_  # Best parameters found
        })

    # Convert the scores list to a DataFrame for easier analysis
    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])


# Call the function to find the best model and display results
best_model_df = find_best_model(X, y)

# Use this to find the location index in the feature set
# This assumes that 'X' is a DataFrame with a column named '2nd Phase Judicial Layout'
location_index = np.where(X.columns == '2nd Phase Judicial Layout')[0][0]


# Function to predict the price based on location, square footage, number of bathrooms, and number of bedrooms
def predict_price(location, sqft, bath, bhk):
    # Find the index of the specified location
    loc_index = np.where(X.columns == location)[0][0]

    # Create an array of zeros to hold the feature values
    x = np.zeros(len(X.columns))  # Initialize feature vector with zeros
    x[0] = sqft  # Square footage
    x[1] = bath  # Number of bathrooms
    x[2] = bhk  # Number of bedrooms

    # If the location index is valid, set the corresponding index in x to 1
    if loc_index >= 0:
        x[loc_index] = 1

    # Use the trained model to predict the price
    return lr_clf.predict([x])[0]  # Return the predicted price


# Example predictions
predicted_price1 = predict_price('1st Phase JP Nagar', 1000, 2, 2)
predicted_price2 = predict_price('Indira Nagar', 1000, 2, 2)

import pickle
with open('BHP/model/banglore_home_prices_model.pickle', 'wb') as f:
    pickle.dump(lr_clf,f)

import json
columns = {
    'data_columns': [col.lower() for col in X.columns]
}
with open('BHP/model/columns.json', 'w') as f:
    f.write(json.dumps(columns))