import os
os.environ['OMP_NUM_THREADS'] = '1'

import myplots as myplt
import time
import warnings
import math
import utils as u
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from matplotlib.backends.backend_pdf import PdfPages
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import pickle
from sklearn import cluster, datasets, mixture
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io


# ----------------------------------------------------------------------
"""
Part 1: 
Evaluation of k-Means over Diverse Datasets: 
In the first task, you will explore how k-Means perform on datasets with diverse structure.
"""

# Fill this function with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

#FUNCTION fit_kmeans - Part B
def fit_kmeans(data, n_clusters):
    
    #Normalize Data
    scaler = StandardScaler()
    scaleddata = scaler.fit_transform(data)
    
    #Train KMeans Model
    kmeans = KMeans(n_clusters=n_clusters, init='random', random_state=42)
    kmeans.fit(scaleddata)
    return kmeans.labels_


def compute():
    answers = {}

    """
    A.	Load the following 5 datasets with 100 samples each: noisy_circles (nc), noisy_moons (nm), blobs with varied variances (bvv), Anisotropicly distributed data (add), blobs (b). Use the parameters from (https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html), with any random state. (with random_state = 42). Not setting the correct random_state will prevent me from checking your results.
    """

    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    # 'nc', 'nm', 'bvv', 'add', 'b'. keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)
    dct = answers["1A: datasets"] = {}
    
    #Generating Datasets
    nc = make_circles(n_samples=100, factor=0.5, noise=0.05, random_state=42)
    nm = make_moons(n_samples=100, noise=0.05, random_state=42)
    bvv = make_blobs(n_samples=100, cluster_std=[1.0, 2.5, 0.5], random_state=42)
    add = make_blobs(n_samples=100, random_state=42)
    b = make_blobs(n_samples=100, random_state=42)
    
    #Transforming Data for Anisotropic Transformation
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    add = (np.dot(add[0], transformation), add[1])
    
    #Storing Generated Datasets in a Dictionary
    answers["1A: datasets"] = {
        'nc': nc,
        'nm' : nm,
        'bvv' : bvv,
        'add' : add,
        'b' : b
    }
    

    """
   B. Write a function called fit_kmeans that takes dataset (before any processing on it), i.e., pair of (data, label) Numpy arrays, and the number of clusters as arguments, and returns the predicted labels from k-means clustering. Use the init='random' argument and make sure to standardize the data (see StandardScaler transform), prior to fitting the KMeans estimator. This is the function you will use in the following questions. 
    """
    
    # dct value:  the `fit_kmeans` function
    dct = answers["1B: fit_kmeans"] = fit_kmeans
    

    """
    C.	Make a big figure (4 rows x 5 columns) of scatter plots (where points are colored by predicted label) with each column corresponding to the datasets generated in part 1.A, and each row being k=[2,3,5,10] different number of clusters. For which datasets does k-means seem to produce correct clusters for (assuming the right number of k is specified) and for which datasets does k-means fail for all values of k? 
    
    Create a pdf of the plots and return in your report. 
    """

    
    #Accessing Datasets from 'answers' Dictionary for Plotting
    datasets = answers["1A: datasets"]
    
    #Specifying K-Values
    k_values = [2, 3, 5, 10]
    # Dictionary to Store KMeans Results
    kmeans_res = {}
    # Iterating through Datasets
    for dataset_name, (data, true_labels) in datasets.items():
        kmeans_ds_res = {}
        for k in k_values:
            # Fitting KMeans Model
            kmeans_labels = fit_kmeans(data, k)
            kmeans_ds_res[k] = kmeans_labels
        # Storing KMeans Results for Current Dataset
        kmeans_res[dataset_name] = ((data, true_labels), kmeans_ds_res)
    
    myplt.plot_part1C(kmeans_res, 'Q1c_kmeans_clusters_evalplots_NG23F.pdf')
    
    # dct value: return a dictionary of one or more abbreviated dataset names (zero or more elements) 
    # and associated k-values with correct clusters.  key abbreviations: 'nc', 'nm', 'bvv', 'add', 'b'. 
    # The values are the list of k for which there is success. Only return datasets where the list of cluster size k is non-empty.
    dct = answers["1C: cluster successes"] = {
        'bvv': [3, 5],
        'b': [2, 3, 5]
    } 

    # dct value: return a list of 0 or more dataset abbreviations (list has zero or more elements, 
    # which are abbreviated dataset names as strings)
    dct = answers["1C: cluster failures"] = ['nc', 'nm', 'add']

    """
    D. Repeat 1.C a few times and comment on which (if any) datasets seem to be sensitive to the choice of initialization for the k=2,3 cases. You do not need to add the additional plots to your report.

    Create a pdf of the plots and return in your report. 
    """
    
    #Specifying New K-Values for Sensitivity Analysis
    new_k_values = [2, 3]
    
    # Dictionary to Store Sensitivity Analysis Results
    sensitivityanalysis = {}
    
    # Iterating through Datasets
    for dataset_name, (data, true_labels) in datasets.items():
        new_kmeans_results_for_dataset = {}
        for new_k in new_k_values:
            all_labels_for_current_new_k = []
            # Running KMeans with Different Initializations
            for init in range(10):
                new_kmeans_labels = fit_kmeans(data, new_k)
                all_labels_for_current_new_k.append(new_kmeans_labels)
            # Storing Results for Current New K
            new_kmeans_results_for_dataset[new_k] = all_labels_for_current_new_k
        # Storing Sensitivity Analysis Results for Current Dataset
        sensitivityanalysis[dataset_name] = ((data, true_labels), new_kmeans_results_for_dataset)
        
    num_datasets = len(sensitivityanalysis)
    num_k_values = len(new_k_values)
    
    fig, axes = plt.subplots(num_k_values, num_datasets, figsize=(15, num_k_values * 3), squeeze=False)
    
    for i, (dataset_name, (data, k_results)) in enumerate(sensitivityanalysis.items()):
        for j, k in enumerate(new_k_values):
            ax = axes[j, i]
            labels_list = k_results[k]
            for labels in labels_list:
                ax.scatter(data[0][:, 0], data[0][:, 1], c=labels, s=1, cmap="viridis", alpha=0.5)
            ax.set_title(f"{dataset_name}, k={k}")
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
            
    plt.tight_layout()
    plt.savefig('Q1d_kmeans_clusters_evalplots_NG23F.pdf')
    plt.close()
    
    # dct value: list of dataset abbreviations
    # Look at your plots, and return your answers.
    # The plot is part of your report, a pdf file name "report.pdf", in your repository.
    dct = answers["1D: datasets sensitive to initialization"] = ["nc","nm"]

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part1.pkl", "wb") as f:
        pickle.dump(answers, f)
