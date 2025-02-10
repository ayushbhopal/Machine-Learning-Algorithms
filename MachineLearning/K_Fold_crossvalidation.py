import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

# Load the digits dataset
digits = load_digits()

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3, random_state=42)

# Define and train the models
lr = LinearRegression()  # Linear Regression is used here
lr.fit(X_train, y_train)
print("Linear Regression Training Score:", lr.score(X_train, y_train))  # Print training score

svm = SVC()  # Support Vector Classifier
svm.fit(X_train, y_train)
print("SVC Training Score:", svm.score(X_train, y_train))  # Print training score

rf = RandomForestClassifier()  # Random Forest Classifier
rf.fit(X_train, y_train)
print("Random Forest Training Score:", rf.score(X_train, y_train))  # Print training score

# -__------------------------------------#-------------------------#_--------------------------#_----------------------------------------

# Define K-Fold cross-validation
kf = KFold(n_splits=3)

# Print train and test indices for each fold
for train_index, test_index in kf.split([1,2,3,4,5,6,7,8,9,10]):
    print("Train indices:", train_index, "Test indices:", test_index)

#-----------------------------------------#------------------------------------------#---------------------------------------------

# Function to get the score of a model
def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)  # Train the model
    return model.score(X_test, y_test)  # Return test score

# Evaluate SVC model using the test set
print("SVC Test Score:", get_score(SVC(), X_train, X_test, y_train, y_test))

# Define StratifiedKFold cross-validation
folds = StratifiedKFold(n_splits=3)

# Lists to store cross-validation scores
scores_l = []
scores_svm = []
scores_rf = []

# Perform cross-validation for each model
for train_index, test_index in kf.split(digits.data):
    X_train, X_test, y_train, y_test = digits.data[train_index], digits.data[test_index], \
                                        digits.target[train_index], digits.target[test_index]

    scores_l.append(get_score(LinearRegression(), X_train, X_test, y_train, y_test))
    scores_svm.append(get_score(SVC(), X_train, X_test, y_train, y_test))
    scores_rf.append(get_score(RandomForestClassifier(), X_train, X_test, y_train, y_test))

# Print cross-validation scores
print("Linear Regression Cross-Validation Scores:", scores_l)
print("SVC Cross-Validation Scores:", scores_svm)
print("Random Forest Cross-Validation Scores:", scores_rf)


from sklearn.model_selection import cross_val_score

cross_val_score(LinearRegression(), digits.data, digits.target)
cross_val_score(RandomForestClassifier(), digits.data, digits.target)
cross_val_score(SVC(), digits.data, digits.target)



# K-Fold and Stratified K-Fold are methods used to check how well a model works by dividing
# the dataset into several parts. K-Fold Cross-Validation splits the data into k parts (or folds).
# It trains the model on k-1 of these parts and tests it on the remaining part, repeating this proces
# s k times. This helps ensure that every part of the data is used for both training and testing.
# However, it doesn't always keep the class balance right, especially if some classes are underrepresented.

# Stratified K-Fold Cross-Validation is similar but includes an extra step:
# it makes sure that each fold has a similar proportion of each class as the original
# dataset. This is especially helpful when dealing with imbalanced classes (like having
# more of one type of class than others). This method ensures that each fold reflects the
# overall class distribution, leading to more reliable and fair performance evaluations.