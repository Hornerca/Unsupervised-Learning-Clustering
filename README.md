# Unsupervised-Learning-Clustering

This analysis explores k-mean clustering algorithm, evaluation methods and limitations on a wine dataset. 

- K-means (sklearn)
- data standardization (StandardScaler): distance-based measurments requries standardized data (mean of zero and SD of 1)
- Evaluation Methods to choose best k
  - Elbow Method: Explores choice of k-clusters based on sum of squared distance (SSE) between data points and assidend clusters's centroids
  - Silhouette Analysis: Explores degree of separation between clusters, where larger separation are better clusters.
  
  
  ## Choosing the best k for the data 
  
  ### Method 1: Elbow Method
  Calculate the sum of squared distance between data points of their respective cluster centroids and choose k based on when the curven starts to flatten out. ( forming an elbow).
  
  Based on the plot below, K=2,3,4 would be good options to explore further. 
  ![Alt Text](https://github.com/Hornerca/Unsupervised-Learning-Clustering/blob/main/Elbow%20Method.png)
  
  ### Method 2: Silhouette Analysis
  Compute silhouette coefficients and choose coefficient that is close to 1 for best clusters. 
  
  Below plots show that 2 clusters has the best average sihouette score. The thickness of the sihouette plot indicates how big the cluser is. 
  
  ![Alt Text](https://github.com/Hornerca/Unsupervised-Learning-Clustering/blob/main/Silhouette%20analysis%20using%20k%20%3D%202.png)
  ![Alt Text](https://github.com/Hornerca/Unsupervised-Learning-Clustering/blob/main/Silhouette%20analysis%20using%20k%20%3D%203.png)
  ![Alt Text](https://github.com/Hornerca/Unsupervised-Learning-Clustering/blob/main/Silhouette%20analysis%20using%20k%20%3D%204.png)
  
  
  # Clustering Data Based on Best k
  K-mean algorithm with different initializations of centroids to explore different results. The title of each plot is the summ of squared distance of each initialization. 
  In practice, we would pick the one with the lowest sum of squared distance. 
 
  ![Alt Text](https://github.com/Hornerca/Unsupervised-Learning-Clustering/blob/main/Centroid_Initializations.png)
  
  
  # Limitations of K-mean Clustering
- The goal of kmeans is to group data points into distinct non-overlapping 
subgroups. 
- It does a very good job when the clusters have a kind of spherical s
hapes. However, it suffers as the geometric shapes of clusters deviates from 
spherical shapes.
- It also doesn’t learn the number of clusters from 
the data and requires it to be pre-defined.
