from __future__ import print_function

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

seed=42
clustermethods = ['KM', 'GM']

def cluster_silh_plot(prefix, clustermethod, range_n_clusters, X, seed=seed):
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
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X.iloc[:, 0], X.iloc[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    if clustermethod == 'KM':
        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for %s clustering on %s data "
                  "with n_clusters = %d\nSilhoutte average %.3f" % (name, prefix.replace("-", " "), n_clusters, silhouette_avg)),
                  fontsize=14, fontweight='bold')
    plt.gcf()
    plt.savefig(prefix+'-'+clustermethod+'-'+str(n_clusters)+'.png')
    plt.close()

    with open(prefix+'-silhscores.csv', "w") as f:
      for line in cluster_scores:
        f.write("%s\n" % (line))

    return cluster_labels, silhouette_avgs

print("Reading credit card data...")
data = pd.read_csv('cctrain.csv')

range_n_clusters = range(2, 11)
X = data.iloc[:,:-1]
label = 'Credit-Card'
plotx = []
ploty = []
for m in clustermethods:
    print ("doing %s..." % (m))
    cluster_labels, silhouette_avgs = cluster_silh_plot(label, m, range_n_clusters, X, seed)
    np.savetxt("%s-%s-cluster-labels.csv" % (label, m), cluster_labels, fmt="%d")
    plotx.append(range_n_clusters)
    ploty.append(silhouette_avgs)
plt.plot(plotx[0], ploty[0], '.-')
plt.plot(plotx[1], ploty[1], '.-')
plt.ylabel("Silhouette Score")
plt.xlabel("n_clusters")
plt.title("Credit Card Data: Silhouette Score")
plt.legend(clustermethods, loc="best")
plt.gcf()
plt.savefig("%s-score.png" % (label))
plt.close()

print("Reading sign language data...")
sldata = pd.read_csv('sltrain.csv')
Xsl = sldata.iloc[:,:-1]

range_n_clusters = np.linspace(5,30,10, dtype='int').tolist()
label = 'Sign-Language'
plotx = []
ploty = []
for m in clustermethods:
    print ("doing %s..." % (m))
    cluster_labels, silhouette_avgs = cluster_silh_plot(label, m, range_n_clusters, Xsl, seed)
    np.savetxt("%s-%s-cluster-labels.csv" % (label, m), cluster_labels, fmt="%d")
    plotx.append(range_n_clusters)
    ploty.append(silhouette_avgs)
plt.plot(plotx[0], ploty[0], '.-')
plt.plot(plotx[1], ploty[1], '.-')
plt.ylabel("Silhouette Score")
plt.xlabel("n_clusters")
plt.title("Sign Language Data: Silhouette Score")
plt.legend(clustermethods, loc="best")
plt.gcf()
plt.savefig("%s-score.png" % (label))
plt.close()
