import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset from the CSV file
df = pd.read_csv("C:\\Users\\ayush\\Downloads\\spam.csv")

# Display descriptive statistics grouped by 'Category'
df.groupby('Category').describe()

# Convert the 'Category' column into a binary format: 1 for 'spam', 0 for 'ham'
df['spam'] = df['Category'].apply(lambda x: 1 if x=='spam' else 0)

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
# X_train: Training data, X_test: Testing data
# y_train: Training labels, y_test: Testing labels
X_train, X_test, y_train, y_test = train_test_split(df.Message, df.spam, test_size=0.2)

from sklearn.feature_extraction.text import CountVectorizer

# Initialize the CountVectorizer
v = CountVectorizer()

# Fit the vectorizer on the training data and transform the training data into a feature matrix
X_train_count = v.fit_transform(X_train.values)

# Convert the sparse matrix to an array and display the first 3 rows
X_train_count.toarray()[:3]

from sklearn.naive_bayes import MultinomialNB

# Initialize the Multinomial Naive Bayes classifier
model = MultinomialNB()

# Train the model using the training data
model.fit(X_train_count, y_train)

# Define new email messages for prediction
emails = [
    'You earned a dividend Account: TFSA Amount: $0.04 Symbol: VRE The cash proceeds will be deposited into your account.',
    'I hope this message finds you well. I would like to confirm that I, Ayush Bhopal, will be switching my schedule with Sanjay Kacha for this month due to the IRCC not yet implementing the 24-hour off-campus work allowance. '
]

# Transform the new email messages into the same feature matrix format
emails_count = v.transform(emails)

# Predict the categories of the new email messages
model.predict(emails_count)

# Transform the test data into the feature matrix
X_test_count = v.transform(X_test)

# Evaluate the model on the test data and print the accuracy score
model.score(X_test_count, y_test)

from sklearn.pipeline import Pipeline

# Create a pipeline that first vectorizes the data and then applies the Multinomial Naive Bayes classifier
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

# Train the pipeline using the training data
clf.fit(X_train, y_train)

# Evaluate the pipeline on the test data and print the accuracy score
clf.score(X_test, y_test)

# Predict the categories of the new email messages using the pipeline
clf.predict(emails)
