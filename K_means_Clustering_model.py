import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Amazon.com Clusturing Model.csv')
X = dataset.iloc[: , 2:4].values

print(X)

from sklearn.cluster import KMeans
wcss = list()
for i in range(1 , 11):
    Kmeans = KMeans(n_clusters = i , init = 'k-means++' , random_state = 21)
    Kmeans.fit(X)
    wcss.append(Kmeans.inertia_)

#Elbow check: to find the optimum no. of clusters
plt.plot(range(1 , 11) , wcss)
plt.title('Elbow Check')
plt.xlabel('No. of clusters')
plt.ylabel('wcss')
plt.show()

Kmeans = KMeans(n_clusters = 4, init = 'k-means++' , random_state = 21)
Kmeans.fit(X)
print(Kmeans.cluster_centers_)
y_pred = Kmeans.predict(X)
print(y_pred)

#visualising
plt.scatter(X[y_pred == 0 , 0] , X[y_pred == 0 , 1] , color = 'red')
plt.scatter(X[y_pred == 1 , 0] , X[y_pred == 1 , 1] , color = 'blue')
plt.scatter(X[y_pred == 2 , 0] , X[y_pred == 2 , 1] , color = 'yellow')
plt.scatter(X[y_pred == 3 , 0] , X[y_pred == 3 , 1] , color = 'green')
plt.scatter(Kmeans.cluster_centers_[: , 0] , Kmeans.cluster_centers_[: , 1] , color = 'black')
plt.title('Cluster of Amazon users')
plt.xlabel('Income')
plt.ylabel('Purchase Rating')
plt.show()
