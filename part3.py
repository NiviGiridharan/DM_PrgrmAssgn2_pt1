import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib.backends.backend_pdf import PdfPages

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

"""
Part 3.	
Hierarchical Clustering: 
Recall from lecture that agglomerative hierarchical clustering is a greedy iterative scheme that creates clusters, i.e., distinct sets of indices of points, by gradually merging the sets based on some cluster dissimilarity (distance) measure. Since each iteration merges a set of indices there are at most n-1 mergers until the all the data points are merged into a single cluster (assuming n is the total points). This merging process of the sets of indices can be illustrated by a tree diagram called a dendrogram. Hence, agglomerative hierarchal clustering can be simply defined as a function that takes in a set of points and outputs the dendrogram.
"""

# Fill this function with code at this location. Do NOT move it.
# Change the arguments and return according to
# the question asked.


def data_index_function(data, I, J):
    # Initialize min_distance with positive infinity
    min_dist = np.inf
    for i in I:
        for j in J:
            # Calculate Euclidean distance between points i and j
            distance = np.linalg.norm(data[i] - data[j])
            # Update min_distance if distance is smaller
            min_dist = min(min_dist, distance)
    return min_dist

def find_merge_iteration(Z, cluster_indices_I, cluster_indices_J, total_points):
    # Initialize cluster membership dictionary with each point belonging to its own cluster
    cluster_membership = {i: {i} for i in range(total_points)}
    # Convert cluster indices to sets for efficient comparison
    cluster_indices_I = {i for i in cluster_indices_I}
    cluster_indices_J = {j for j in cluster_indices_J}
    
    # Iterate through the linkage matrix
    for iteration, (cluster1, cluster2, _, _) in enumerate(Z):
        new_cluster = cluster_membership[cluster1].union(cluster_membership[cluster2])
        # Check if the specified clusters are merged
        if cluster_indices_I.issubset(new_cluster) and cluster_indices_J.issubset(new_cluster):
            return iteration
        
        # Update cluster membership
        cluster_membership[total_points + iteration] = new_cluster
        del cluster_membership[cluster1]
        del cluster_membership[cluster2]

    return -1

def compute():
    answers = {}

    """
    A.	Load the provided dataset “hierachal_toy_data.mat” using the scipy.io.loadmat function.
    """

    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'hierarchical_toy_data.mat')
    
    toy_data = io.loadmat(file_path)
    
    # return value of scipy.io.loadmat()
    answers["3A: toy data"] = {
        "X": toy_data["X"],
        "y": toy_data["y"].squeeze()
    }

    """
    B.	Create a linkage matrix Z, and plot a dendrogram using the scipy.hierarchy.linkage and scipy.hierachy.dendrogram functions, with “single” linkage.
    """


    # Extract the feature matrix from the loaded toy data
    toy_data_X = toy_data["X"]
    Z = linkage(toy_data_X, 'single')
    
    # Generate a dendrogram plot and save it as a PDF file
    with PdfPages(os.path.join(script_dir, 'Q3b_Dendrogram_NG23F.pdf')) as pdf:
        plt.figure(figsize=(10, 7))
        plt.title('Hierarchical Clustering Dendrogram (Single Linkage)')
        plt.xlabel('Sample index')
        plt.ylabel('Distance')
        # Generate the dendrogram and store its dictionary representation
        dendro_dict = dendrogram(Z)
        pdf.savefig()
        plt.close()
    
    # Store the linkage matrix Z in the answers dictionary
    answers["3B: linkage"] = Z

    # Store the dictionary representation of the dendrogram in the answers dictionary
    answers["3B: dendogram"] = dendro_dict

    """
    C.	Consider the merger of the cluster corresponding to points with index sets {I={8,2,13}} J={1,9}}. At what iteration (starting from 0) were these clusters merged? That is, what row does the merger of A correspond to in the linkage matrix Z? The rows count from 0. 
    """

    # Specify the index sets for the clusters to be merged
    I = [8, 2, 13]
    J = [1, 9]
    
    # Find the iteration number when the specified clusters are merged
    merge_iteration = find_merge_iteration(Z, I, J, len(toy_data_X))
    
    # Store the merge iteration number in the answers dictionary
    answers["3C: iteration"] = merge_iteration

    """
    D.	Write a function that takes the data and the two index sets {I,J} above, and returns the dissimilarity given by single link clustering using the Euclidian distance metric. The function should output the same value as the 3rd column of the row found in problem 2.C.
    """
    
    # Answer type: a function defined above
    answers["3D: function"] = data_index_function

    """
    E.	In the actual algorithm, deciding which clusters to merge should consider all of the available clusters at each iteration. List all the clusters as index sets, using a list of lists, 
    e.g., [{0,1,2},{3,4},{5},{6},…],  that were available when the two clusters in part 2.D were merged.
    """

    # Initialize clusters with each data point forming its own cluster
    clusters = [{i} for i in range(len(toy_data_X))]
    merge_step = 4
    new_clusters = []
    
    for i in range(merge_step + 1):
        idx1, idx2 = int(Z[i, 0]), int(Z[i, 1])
        # Form new cluster by combining clusters at idx1 and idx2
        new_cluster = clusters[idx1].union(clusters[idx2])
        new_clusters.append(new_cluster)
        clusters.append(new_cluster)

    # Convert new_clusters to list of lists    
    final_clusters = [list(cluster) for cluster in new_clusters if cluster]

    # List the clusters as a list of lists
    answers["3E: clusters"] = final_clusters

    """
    F.	Single linked clustering is often criticized as producing clusters where “the rich get richer”, that is, where one cluster is continuously merging with all available points. Does your dendrogram illustrate this phenomenon?
    """

    # Answer type: string. Insert your explanation as a string.
    answers["3F: rich get richer"] = "The dendrogram illustrates the 'rich get richer' phenomenon typical in single linkage clustering. This can be observed when more points or smaller clusters are regularly added to bigger clusters, resulting in an unbalanced distribution where a few or one cluster dominates. The elongated chains in the dendrogram, where individual points or small clusters join larger clusters, highlight this pattern."

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part3.pkl", "wb") as f:
        pickle.dump(answers, f)
