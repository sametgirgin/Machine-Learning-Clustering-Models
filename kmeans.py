# K-Means Clustering 

#1 Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#2 Importing the mall dataset
dataset= pd.read_csv('Mall_Customers.csv')
#Select the annual income and the spending score columns  
X=dataset.iloc[:, [3,4]].values

#3 Using the elbow method to find out the optimal number of clusters. 
#KMeans class from the sklearn library. 
from sklearn.cluster import KMeans
wcss=[]
#this loop will fit the k-means algorithm to our data and 
#second we will compute the within cluster sum of squares and appended to our wcss list.
for i in range(1,11): 
     kmeans = KMeans(n_clusters=i, init ='k-means++',max_iter=300,n_init=10,random_state=0 )
#i above is between 1-10 numbers. init parameter is the random initialization method  
#we select kmeans++ method. max_iter parameter the maximum number of iterations there can be to 
#find the final clusters when the K-meands algorithm is running. we enter the default value of 300
#the next parameter is n_init which is the number of times the K_means algorithm will be run with
#different initial centroid.
     kmeans.fit(X)
#kmeans algorithm fits to the X dataset
     wcss.append(kmeans.inertia_)
#kmeans inertia_ attribute is:  Sum of squared distances of samples to their closest cluster center.

#4.Plot the elbow graph
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#5 According to the Elbow graph we deterrmine the clusters number as 5. Applying k-means 
#algorithm to the X dataset. 
kmeans = KMeans(n_clusters=5, init ='k-means++',max_iter=300,n_init=10,random_state=0 )
# We are going to use the fit predict method that returns for each observation which 
#cluster it belongs to. The cluster to which client belongs and it will return this 
#cluster numbers into a single vector that is  called y K-means
y_kmeans = kmeans.fit_predict(X)

#6 Visualising the clusters

plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=100, c='red', label ='Cluster 1')
plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], s=100, c='blue', label ='Cluster 2')
plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], s=100, c='green', label ='Cluster 3')
plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], s=100, c='cyan', label ='Cluster 4')
plt.scatter(X[y_kmeans==4, 0], X[y_kmeans==4, 1], s=100, c='magenta', label ='Cluster 5')

#Plot the centroid. This time we're going to use the cluster centres  attribute that 
#returns here the coordinates of the centroid.

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label ='Centroids')
plt.title('Clusters of Customers')
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score(1-100')
plt.show()
