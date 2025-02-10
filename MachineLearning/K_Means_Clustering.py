import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# Load the dataset
df = pd.read_csv("C:\\Users\\ayush\\Downloads\\income.csv")
df

# Visualize the data
# Scatter plot of Age vs. Income to get an initial view of the distribution
plt.scatter(df['Age'], df['Income($)'])
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.title('Scatter plot of Age vs. Income')
plt.show(block=True)

# Initialize the KMeans clustering algorithm with 3 clusters
km = KMeans(n_clusters=3)
# Fit the model and predict the cluster labels for each data point
y_predicted = km.fit_predict(df[['Age', 'Income($)']])
y_predicted

# Uncomment the following lines to visualize the clusters before scaling

# df['cluster'] = y_predicted
#
# df1 = df[df.cluster==0]  # Data for cluster 0
# df2 = df[df.cluster==1]  # Data for cluster 1
# df3 = df[df.cluster==2]  # Data for cluster 2
#
# # Plot the clusters with different colors
# plt.scatter(df1.Age, df1['Income($)'], color='green', label='Cluster 0')
# plt.scatter(df2.Age, df2['Income($)'], color='red', label='Cluster 1')
# plt.scatter(df3.Age, df3['Income($)'], color='black', label='Cluster 2')
# plt.xlabel("Age")
# plt.ylabel("Income($)")
# plt.legend()
# plt.title('Clusters before Scaling')
# plt.show(block=True)

# Apply Min-Max Scaling to normalize the features
# This scales the features to be between 0 and 1
scaler = MinMaxScaler()
df[['Age', 'Income($)']] = scaler.fit_transform(df[['Age', 'Income($)']])
print(df.head())

# Reapply KMeans clustering on the scaled data
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Age', "Income($)"]])
# Add the cluster labels to the dataframe
df['cluster'] = y_predicted

# Extract data points for each cluster
df1 = df[df.cluster==0]  # Data for cluster 0
df2 = df[df.cluster==1]  # Data for cluster 1
df3 = df[df.cluster==2]  # Data for cluster 2

# Plot the clusters with different colors and centroids
plt.scatter(df1.Age, df1['Income($)'], color='green', label='Cluster 0')
plt.scatter(df2.Age, df2['Income($)'], color='red', label='Cluster 1')
plt.scatter(df3.Age, df3['Income($)'], color='black', label='Cluster 2')
# Plot the centroids of the clusters
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color='purple', marker='*', label='Centroids')
plt.xlabel("Age")
plt.ylabel("Income($)")
plt.legend()
plt.title('Clusters after Scaling')
plt.show(block=True)

# Determine the optimal number of clusters using the Elbow method
krange = range(1,10)  # Test different numbers of clusters
SSE = []  # Sum of squared errors for each k

for k in krange:
    km = KMeans(n_clusters=k)
    km.fit(df[['Age', "Income($)"]])
    SSE.append(km.inertia_)  # Inertia: Sum of squared distances to nearest cluster center

# Plot the Elbow curve to visualize the optimal number of clusters
plt.xlabel("K")
plt.ylabel("Sum of Squared Error (SSE)")
plt.plot(krange, SSE, marker='o')
plt.title('Elbow Method for Optimal K')
plt.show(block=True)
