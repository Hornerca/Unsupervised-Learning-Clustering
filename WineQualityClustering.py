#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 14:47:32 2021

@author: christine.horner
"""

# Modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import pandas as pd
import seaborn as sns
from sklearn.datasets.samples_generator import (make_blobs,
                                                make_circles,
                                                make_moons)
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score



# https://www.kaggle.com/xvivancos/tutorial-clustering-wines-with-k-means
# https://data.world/food/wine-quality

red=pd.read_csv('winequality-red.csv')
white=pd.read_csv('winequality-white.csv')
winedata=pd.concat([red, white])

df=winedata.drop(columns=['quality'])
df_2=df.loc[:, ['alcohol', 'density']]
y=winedata['quality']

# k-means clustering
# kmeans = KMeans(n_clusters=2, random_state=0).fit(data)




"""
k mean clustering:
    
Since clustering algorithms including kmeans use distance-based measurements 
to determine the similarity between data points, it’s recommended to standardize 
the data to have a mean of zero and a standard deviation of one since almost 
always the features in any dataset would have different units of measurements 
such as age vs income.


it’s recommended to run the algorithm using different initializations of 
centroids and pick the results of the run that that yielded the lower sum 
of squared distance.

https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a


model assumptions:
The goal of kmeans is to group data points into distinct non-overlapping 
subgroups. It does a very good job when the clusters have a kind of spherical s
hapes. However, it suffers as the geometric shapes of clusters deviates from 
spherical shapes. Moreover, it also doesn’t learn the number of clusters from 
the data and requires it to be pre-defined.
"""


# Standardize the data
X_std = StandardScaler().fit_transform(df_2)



#%% Evaluation methods to Choose K Clusters
# Run the Kmeans algorithm and get the index of data points clusters
"""
Elbow method gives us an idea on what a good k number of clusters would be 
based on the sum of squared distance (SSE) between data points and their 
assigned clusters’ centroids. We pick k at the spot where SSE starts to 
flatten out and forming an elbow.
"""


sse = []
list_k = list(range(1, 10))

for k in list_k:
    km = KMeans(n_clusters=k)
    km.fit(X_std)
    sse.append(km.inertia_)

# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse, '-o')
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance')
plt.savefig('Elbow Method')

#%%
"""
Silhouette analysis can be used to determine the degree of separation between clusters.

"""
for i, k in enumerate([2, 3, 4]):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    
    # Run the Kmeans algorithm
    km = KMeans(n_clusters=k)
    labels = km.fit_predict(X_std)
    centroids = km.cluster_centers_

    # Get silhouette samples
    silhouette_vals = silhouette_samples(X_std, labels)

    # Silhouette plot
    y_ticks = []
    y_lower, y_upper = 0, 0
    for i, cluster in enumerate(np.unique(labels)):
        cluster_silhouette_vals = silhouette_vals[labels == cluster]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        ax1.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
        ax1.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))
        y_lower += len(cluster_silhouette_vals)

    # Get the average silhouette score and plot it
    avg_score = np.mean(silhouette_vals)
    ax1.axvline(avg_score, linestyle='--', linewidth=2, color='green')
    ax1.set_yticks([])
    ax1.set_xlim([-0.1, 1])
    ax1.set_xlabel('Silhouette coefficient values')
    ax1.set_ylabel('Cluster labels')
    ax1.set_title('Silhouette plot for the various clusters', y=1.02);
    
    # Scatter plot of data colored with labels
    ax2.scatter(X_std[:, 0], X_std[:, 1], c=labels)
    ax2.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='r', s=250)
    ax2.set_xlim([-2, 2])
    ax2.set_xlim([-2, 2])
    ax2.set_xlabel('Eruption time in mins')
    ax2.set_ylabel('Waiting time to next eruption')
    ax2.set_title('Visualization of clustered data', y=1.02)
    ax2.set_aspect('equal')
    plt.tight_layout()
    plt.suptitle(f'Silhouette analysis using k = {k}',
                 fontsize=16, fontweight='semibold', y=1.05);
    
    plt.savefig(f'Silhouette analysis using k = {k}.png')

#%% Different Centroid Initilizations to 

n_iter = 4
fig, ax = plt.subplots(2, 2)
ax = np.ravel(ax)
centers = []
for i in range(n_iter):
    # Run local implementation of kmeans
    km = KMeans(n_clusters=2,
                max_iter=3,
                random_state=np.random.randint(0, 1000, size=1)[0])
    km.fit(X_std)
    centroids = km.cluster_centers_
    centers.append(centroids)
    ax[i].scatter(X_std[km.labels_ == 0, 0], X_std[km.labels_ == 0, 1],
                  c='green', label='cluster 1')
    ax[i].scatter(X_std[km.labels_ == 1, 0], X_std[km.labels_ == 1, 1],
                  c='blue', label='cluster 2')
    ax[i].scatter(centroids[:, 0], centroids[:, 1],
                  c='r', marker='*', s=300, label='centroid')
    ax[i].set_xlim([-2, 2])
    ax[i].set_ylim([-2, 2])
    ax[i].legend(loc='lower right')
    ax[i].set_title(f'{km.inertia_:.4f}')
    ax[i].set_aspect('equal')
fig.suptitle('Different Initializations of Centroids')
plt.tight_layout();
plt.savefig('Centroid_Initializations.png')
