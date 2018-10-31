from __future__ import print_function

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score, f1_score, accuracy_score
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from common import *
import numpy as np
import pandas as pd

seed=42
clustermethods = ['KM', 'GM']

def cluster_silh_plot(prefix, clustermethod, range_n_clusters, X, plotdim, seed=seed):
    if clustermethod not in clustermethods:
        print("Invalid cluster method %s" % (clustermethod))
        return None
    
    silhouette_avgs = []
    sample_silhouette_nvalues = []
    cluster_nlabels = []
    clusterers = []
    cluster_scores = ["method,nclusters,score"]
    for n_clusters in range_n_clusters:
        # Initialize the clusterer with n_clusters value and a random generator
        # seed for reproducibility.
        if clustermethod == 'GM':
            name = 'GaussianMixture'
            clusterer = GaussianMixture(n_components=n_clusters, random_state=seed)        
        if clustermethod == 'KM':
            name = 'KMeans'
            clusterer = KMeans(n_clusters=n_clusters, random_state=seed)

        clusterers.append(clusterer)

        # Predict cluster labels
        cluster_labels = clusterer.fit(X).predict(X)
        cluster_nlabels.append(cluster_labels)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_avgs.append(silhouette_avg)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)
        cluster_scores.append("%s,%d,%.10f" % (clustermethod,n_clusters,silhouette_avg))

        # Compute the silhouette scores for each sample
        sample_silhouette_nvalues.append(silhouette_samples(X, cluster_labels))

    highest_score = -1
    n_clusters = None
    cluster_labels = None
    sample_silhouette_values = None
    silhouette_avg = None
    clusterer = None

    for i, v in enumerate(silhouette_avgs):
      if v > highest_score:
        n_clusters = range_n_clusters[i]
        silhouette_avg = silhouette_avgs[i]
        sample_silhouette_values = sample_silhouette_nvalues[i]
        cluster_labels = cluster_nlabels[i]
        clusterer = clusterers[i]
        highest_score = v

    print("highest silhoutte score = %.10f" % (silhouette_avg))
    print("n_clusters with highest score = %d" % (n_clusters))
    print("plotting...")

    plot_clusters(prefix, clustermethod, name, X, cluster_labels, n_clusters, plotdim)
    plot_silh(prefix, clustermethod, name, n_clusters, X, cluster_labels, clusterer, silhouette_avg, sample_silhouette_values)

    with open(prefix.replace(" ", "-")+'-silhscores.csv', "w") as f:
      for line in cluster_scores:
        f.write("%s\n" % (line))

    return cluster_labels, silhouette_avgs

scorers = [[f1_score, accuracy_score], ['f1', 'accuracy']]
#plotdata = []
#xlabels = []
for id in [1,2]:
    X, y, label, _, range_n_clusters = getDataset(id)
    #xlabels.append('%s\n(%s)' % (label, scorers[1][id-1]))
    plotx = []
    ploty = []
    plotcomponents = [[(1, 0, 2), (1, 0, 2)],[(91, 397, 611), (0, 1, 678)]]
    #scores = []
    for i, m in enumerate(clustermethods):
        print ("doing %s..." % (m))
        cluster_labels, silhouette_avgs = cluster_silh_plot(label, m, range_n_clusters, X, plotcomponents[id-1][i])
        np.savetxt("%s-%s-cluster-labels.csv" % (label.replace(" ", "-"), m), cluster_labels, fmt="%d")
        plotx.append(range_n_clusters)
        ploty.append(silhouette_avgs)
        if (np.unique(cluster_labels).shape[0] == np.unique(y).shape[0]):
            score = scorers[0][id-1](y, cluster_labels)
            print("method %s, scorer %s, score %.3f" % (m, scorers[1][i], score))
        #scores.append(score)
    #plotdata.append(scores)
    plot_silhscores(label, plotx, ploty, clustermethods)
#title = "Accuracy/F1 Score of Clustering Algorithms"
#plot_2bar(plotdata[0], plotdata[1], clustermethods, xlabels, [0, 1], 'f1/accuracy score', title, 'q1score.png')