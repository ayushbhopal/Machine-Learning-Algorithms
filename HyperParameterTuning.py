import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

# Load the Iris dataset
iris = load_iris()

# Create a DataFrame from the Iris dataset
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['flower'] = iris.target

# Map numerical target values to their corresponding flower names
df['flower'] = df['flower'].apply(lambda x: iris.target_names[x])

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Initialize the GridSearchCV object
# `SVC(gamma='auto')` is the model to tune
# The parameters to search over are 'C' and 'kernel'
# `cv=5` means 5-fold cross-validation
# `return_train_score=False` omits train scores in results
clf = GridSearchCV(SVC(gamma='auto'), {
    'C': [1, 10, 20],
    'kernel': ['rbf', 'linear']
}, cv=5, return_train_score=False)

# Fit the GridSearchCV object to the Iris dataset
# This will perform a grid search over specified parameters and find the best model
clf.fit(iris.data, iris.target)

# Display the cross-validation results
# This DataFrame includes the parameter combinations and corresponding test scores
df = pd.DataFrame(clf.cv_results_)
df[['param_C', 'param_kernel', 'mean_test_score']]

# Display the best score and parameters found by GridSearchCV
print("Best Score:", clf.best_score_)
print("Best Parameters:", clf.best_params_)

# Initialize the RandomizedSearchCV object
# RandomizedSearchCV performs a random search over a specified number of parameter combinations
# `n_iter=2` specifies the number of random combinations to try
rs = RandomizedSearchCV(SVC(gamma='auto'), {
    'C': [1, 10, 15, 20],
    'kernel': ['rbf', 'linear', 'sigmoid']
},
                        cv=5,
                        return_train_score=False,
                        n_iter=2,
                        random_state=42)  # Added random_state for reproducibility

# Fit the RandomizedSearchCV object to the Iris dataset
# This will perform a random search over specified parameters and find the best model
rs.fit(iris.data, iris.target)

# Display the randomized search results
# This DataFrame includes the parameter combinations and corresponding test scores
pd.DataFrame(rs.cv_results_)[['param_C', 'param_kernel', 'mean_test_score']]


#--------------------------------------------#---------------------------------------------#-----------------------------------


from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

# Define the models and their hyperparameter grids
model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params': {
            'C': [1, 10, 15, 20],
            'kernel': ['rbf', 'sigmoid', 'linear']
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [1, 5, 10]  # Corrected 'n_estimator' to 'n_estimators'
        }
    },
    'logistic_regression': {
        'model': LogisticRegression(solver='liblinear', multi_class='auto'),
        'params': {
            'C': [1, 5, 7, 10]
        }
    }
}

# Initialize a list to store the results
scores = []

# Iterate through the model parameters and perform GridSearchCV
for model_name, mp in model_params.items():
    clf = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(iris.data, iris.target)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })

df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
df



# GridSearchCV and RandomizedSearchCV are methods for finding the best settings
# for a machine learning model. GridSearchCV tests every possible combination of
# settings you provide to find the best one. It’s like trying out every recipe to
# see which one tastes best. RandomizedSearchCV takes a quicker approach by randomly
# picking some combinations to test, rather than trying every single one. It’s like
# sampling a few recipes from a large cookbook instead of trying them all. Both methods
# use cross-validation to check how well each set of settings works, and then pick the best one.