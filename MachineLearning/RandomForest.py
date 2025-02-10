import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import seaborn as sns
digits = load_digits()
dir(digits)

df = pd.DataFrame(digits.data)
df
df['target'] = digits.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop(['target'], axis=1), df.target, test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
model.score(X_test, y_test)

y_predicted = model.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
cm

plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True)
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show(block=True)



# ABOUT THE CONCEPT OF RANDOM FOREST AND CONFUSION MATRIX
#
# **Random Forest**:
# - Random Forest is an ensemble learning method that uses multiple decision trees to make predictions.
# - It builds several decision trees during training and combines their outputs to improve accuracy and control overfitting.
# - Each tree in the forest is trained on a random subset of the data and features, which helps in creating a diverse set of models.
# - The final prediction is made by aggregating the predictions from all individual trees (e.g., by majority voting for classification).
#
# **Confusion Matrix**:
# - A confusion matrix is a tool used to evaluate the performance of a classification model.
# - It shows the counts of true positive, true negative, false positive, and false negative predictions.
# - This matrix helps in understanding how well the model is performing across different classes and where it is making errors.
# - The matrix is visualized using a heatmap for easier interpretation of the results.
