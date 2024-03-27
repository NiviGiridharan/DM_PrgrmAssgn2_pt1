import os
os.environ['OMP_NUM_THREADS'] = '1'

from pprint import pprint

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

import myplots as myplt
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage  #

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

# ----------------------------------------------------------------------
"""
Part 2
Comparison of Clustering Evaluation Metrics: 
In this task you will explore different methods to find a good value for k
"""

# Fill this function with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_kmeans(data, n_clusters):
    # Initializing KMeans Clustering with specified number of clusters and random state
    kmeans = KMeans(n_clusters=n_clusters, random_state=12)
    kmeans.fit(data)
    # Calculating Sum of Squared Errors (SSE) for the clustering
    sse = 0
    for abc, point in enumerate(data):
        center = kmeans.cluster_centers_[kmeans.labels_[abc]]
        dist = np.linalg.norm(point - center)
        sse += dist ** 2
    return sse

def fit_kmeans_inertia(data, n_clusters):
    # Initializing KMeans Clustering with specified number of clusters and random state
    kmeans = KMeans(n_clusters=n_clusters, random_state=12)
    # Fitting the KMeans model to the data
    kmeans.fit(data)
    # Getting the inertia (sum of squared distances to the closest cluster center) of the clustering
    inertia = kmeans.inertia_
    return inertia

def compute():
    # ---------------------
    answers = {}

    """
    A.	Call the make_blobs function with following parameters :(center_box=(-20,20), n_samples=20, centers=5, random_state=12).
    """

    #Invoke the Function with Given Parameters
    X, y, centers = make_blobs(n_samples=20, centers=5, center_box=(-20, 20), random_state=12, return_centers=True)
    
    # dct: return value from the make_blobs function in sklearn, expressed as a list of three numpy arrays
    dct = answers["2A: blob"] = [X, y, centers]

    """
    B. Modify the fit_kmeans function to return the SSE (see Equations 8.1 and 8.2 in the book).
    """

    # dct value: the `fit_kmeans` function
    dct = answers["2B: fit_kmeans"] = fit_kmeans

    """
    C.	Plot the SSE as a function of k for k=1,2,….,8, and choose the optimal k based on the elbow method.
    """

    #Specifying the Range of K Values
    k_values = range(1, 9)
    
    # Initialize an Empty List to Store SSE Values
    sse_values = []
    
    # Calculate SSE for Each K Value
    for k in k_values:
        sse = fit_kmeans(X, k)
        sse_values.append(sse)
        
    # Plot SSE for Each K Value
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, sse_values, marker='o')
    plt.title('Scree Plot')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Sum of Squared Errors')
    plt.xticks(k_values)
    plt.grid(True)
    plt.savefig('Q2c_SSEplot_NG23F.pdf')
    plt.show()
    
    # Determine the Optimal K Value
    optimal_k_value = 3
    
    # dct value: a list of tuples, e.g., [[0, 100.], [1, 200.]]
    # Each tuple is a (k, SSE) pair
    dct = answers["2C: SSE plot"] = [[k, round(sse, 2)] for k, sse in zip(k_values, sse_values)]

    """
    D.	Repeat part 2.C for inertia (note this is an attribute in the kmeans estimator called _inertia). Do the optimal k’s agree?
    """

    #Specifying the Range of K Values
    using_inertia_k_values = range(1, 9)
    
    # Initialize an Empty List to Store SSE Values
    using_inertia_sse_values = []
    
    # Calculate SSE for Each K Value
    for inertia_k in using_inertia_k_values:
        inertia_sse = fit_kmeans_inertia(X, inertia_k)
        using_inertia_sse_values.append(inertia_sse)
        
    # Plot SSE for Each K Value
    plt.figure(figsize=(10, 6))
    plt.plot(using_inertia_k_values, using_inertia_sse_values, marker='o')
    plt.title('Determining Optimal Number of Clusters using Inertia')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.xticks(using_inertia_k_values)
    plt.grid(True)
    plt.savefig('Q2d_Inretiaplot_NG23F.pdf')
    plt.show()
    
    inertia_optimal_k_value = 3
    
    # dct value has the same structure as in 2C
    dct = answers["2D: inertia plot"] = [[inertia_k, round(inertia_sse, 2)] for inertia_k, inertia_sse in zip(using_inertia_k_values, using_inertia_sse_values)]

    # dct value should be a string, e.g., "yes" or "no"
    dct = answers["2D: do ks agree?"] = "yes"

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part2.pkl", "wb") as f:
        pickle.dump(answers, f)
