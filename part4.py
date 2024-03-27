import os
os.environ['OMP_NUM_THREADS'] = '1'

import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib.backends.backend_pdf import PdfPages

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

"""
Part 4.	
Evaluation of Hierarchical Clustering over Diverse Datasets:
In this task, you will explore hierarchical clustering over different datasets. You will also evaluate different ways to merge clusters and good ways to find the cut-off point for breaking the dendrogram.
"""

# Fill these two functions with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_hierarchical_cluster(data, linkage_type, n_clusters):
    
    # Standardize the input data
    scaler = StandardScaler()
    scaleddata = scaler.fit_transform(data)
    
    ## Fit Agglomerative Clustering model with specified linkage type and number of clusters
    model = AgglomerativeClustering(linkage=linkage_type, n_clusters=n_clusters)
    model.fit(scaleddata)
    return model.labels_


def fit_modified(data, method):
    
    # Calculate the linkage matrix using the specified method
    XX = linkage(data, method=method)
    
    # Find the largest gap in distances between consecutive merges
    distances = XX[:, 2]
    distance_differences = np.diff(distances)
    max_diff_idx = np.argmax(distance_differences)
    cut_off_distance = (distances[max_diff_idx] + distances[max_diff_idx + 1]) / 2
    
    return XX, cut_off_distance


def compute():
    answers = {}

    """
    A.	Repeat parts 1.A and 1.B with hierarchical clustering. That is, write a function called fit_hierarchical_cluster (or something similar) that takes the dataset, the linkage type and the number of clusters, that trains an AgglomerativeClustering sklearn estimator and returns the label predictions. Apply the same standardization as in part 1.B. Use the default distance metric (euclidean) and the default linkage (ward).
    """

    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    # keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)
    dct = answers["4A: datasets"] = {}
    
    # Generate various datasets for clustering analysis
    nc = make_circles(n_samples=100, factor=0.5, noise=0.05, random_state=42)
    nm = make_moons(n_samples=100, noise=0.05, random_state=42)
    bvv = make_blobs(n_samples=100, cluster_std=[1.0, 2.5, 0.5], random_state=42)
    add = make_blobs(n_samples=100, random_state=42)
    b = make_blobs(n_samples=100, random_state=42)
    
    # Apply transformation to introduce anisotropy in the data
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    add = (np.dot(add[0], transformation), add[1])
    # Store the generated datasets in the answers dictionary
    answers["4A: datasets"] = {
        'nc': nc,
        'nm' : nm,
        'bvv' : bvv,
        'add' : add,
        'b' : b
    }

    # dct value:  the `fit_hierarchical_cluster` function
    dct = answers["4A: fit_hierarchical_cluster"] = fit_hierarchical_cluster

    """
    B.	Apply your function from 4.A and make a plot similar to 1.C with the four linkage types (single, complete, ward, centroid: rows in the figure), and use 2 clusters for all runs. Compare the results to problem 1, specifically, are there any datasets that are now correctly clustered that k-means could not handle?

    Create a pdf of the plots and return in your report. 
    """
    # Access datasets from the answers dictionary for plotting
    datasets = answers["4A: datasets"]
    
    # Specify the linkage types to use
    linkage_types = ['single', 'complete', 'ward', 'average']
    # Dictionary to store hierarchical clustering results
    hierarchical_clustering_results = {}
    
    fig, axs = plt.subplots(len(datasets), 4, figsize=(20, 15))
    
    for i, (dataset_name, (X, y)) in enumerate(datasets.items()):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        for j, linkage in enumerate(linkage_types):
            # Fit hierarchical clustering and get cluster labels
            hierarchical_clustering_labels = fit_hierarchical_cluster(X_scaled, linkage, 2)
            axs[i, j].scatter(X_scaled[:, 0], X_scaled[:, 1], c=hierarchical_clustering_labels, s=50, cmap='viridis', edgecolor='k')
            axs[i, j].set_title(f'{dataset_name} - {linkage}')
            
            if dataset_name not in hierarchical_clustering_results:
                hierarchical_clustering_results[dataset_name] = {}
            hierarchical_clustering_results[dataset_name][linkage] = hierarchical_clustering_labels
    
    plt.tight_layout()
    plt.savefig('Q4b_Hierarchical_clusteringplots_NG23F.pdf')
    
    # dct value: list of dataset abbreviations (see 1.C)
    dct = answers["4B: cluster successes"] = ["nc", "nm"]

    """
    C.	There are essentially two main ways to find the cut-off point for breaking the diagram: specifying the number of clusters and specifying a maximum distance. The latter is challenging to optimize for without knowing and/or directly visualizing the dendrogram, however, sometimes simple heuristics can work well. The main idea is that since the merging of big clusters usually happens when distances increase, we can assume that a large distance change between clusters means that they should stay distinct. Modify the function from part 1.A to calculate a cut-off distance before classification. Specifically, estimate the cut-off distance as the maximum rate of change of the distance between successive cluster merges (you can use the scipy.hierarchy.linkage function to calculate the linkage matrix with distances). Apply this technique to all the datasets and make a plot similar to part 4.B.
    
    Create a pdf of the plots and return in your report. 
    """

    # Access datasets from the answers dictionary for plotting
    datasets = answers["4A: datasets"]
    # Specify the linkage methods to use
    linkage_methods = ['ward', 'complete', 'average', 'single']
    
    with PdfPages('Q4c_Hierarchical_clusteringplots_NG23F.pdf') as pdf:
        for dataset_name, (X, y) in datasets.items():
            
            # Scale the dataset
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            # Create subplots for each linkage method
            fig, axs = plt.subplots(1, len(linkage_methods), figsize=(20, 5))
            fig.suptitle(f'Clustering for {dataset_name}', fontsize=16)
            # Iterate over each linkage method
            for j, linkage_method in enumerate(linkage_methods):
                # Fit hierarchical clustering and obtain clusters
                Z, cut_off_distance = fit_modified(X_scaled, linkage_method)
                clusters = fcluster(Z, cut_off_distance, criterion='distance')
                axs[j].scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', edgecolor='k', s=50)
                axs[j].set_title(f'{dataset_name} - {linkage_method}')
                axs[j].set_xlabel('Feat1')
                axs[j].set_ylabel('Feat2')
                
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig(fig)
            plt.close()
    
    # dct is the function described above in 4.C
    dct = answers["4C: modified function"] = fit_modified

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part4.pkl", "wb") as f:
        pickle.dump(answers, f)
