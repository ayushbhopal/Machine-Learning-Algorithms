import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import tree

iris = load_iris()
dir(iris)
df = pd.DataFrame(iris.data, columns = iris.feature_names)

df['target'] = iris.target
df['flower_names'] = df.target.apply(lambda  x: iris.target_names[x])

from sklearn.model_selection import train_test_split
X= df.drop(['target', 'flower_names'], axis=1)
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.model_selection import StratifiedKFold
folds = StratifiedKFold(n_splits=5)

from sklearn.model_selection import cross_val_score
cross_val_score(LogisticRegression(), X, y)
cross_val_score(tree.DecisionTreeClassifier(), X, y)
cross_val_score(RandomForestClassifier(), X, y)
cross_val_score(SVC(), X, y)