# -------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #5
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

# importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

df = pd.read_csv('training_data.csv', sep=',', header=None)  # reading the data by using Pandas library

# assign your training data to X_training feature matrix
X_training = np.array(df.values)

# run kmeans testing different k values from 2 until 20 clusters
# Use:  kmeans = KMeans(n_clusters=k, random_state=0)
#      kmeans.fit(X_training)

silhouettes = []

for k in range(2, 21):
    # Clearing the Screen
    print("Running k=" + k.__str__())
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_training)

    # for each k, calculate the silhouette_coefficient by using: silhouette_score(X_training, kmeans.labels_)
    # find which k maximizes the silhouette_coefficient
    silhouette_coefficient = silhouette_score(X_training, kmeans.labels_)
    silhouettes.append(silhouette_coefficient)

# plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
plt.scatter(range(2, 21), silhouettes)
plt.show()

# reading the test data (clusters) by using Pandas library
df = pd.read_csv('testing_data.csv', sep=',', header=None)  # reading the data by using Pandas library

# assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
# do this: np.array(df.values).reshape(1,<number of samples>)[0]
X_test = np.array(df.values)
labels = X_test.reshape(1, 3823)[0]

# kmeans = KMeans(n_clusters=10, random_state=0)
# kmeans.fit(X_test)

# Calculate and print the Homogeneity of this kmeans clustering
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())
# --> add your Python code here
