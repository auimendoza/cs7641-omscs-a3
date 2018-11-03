from __future__ import print_function

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy.linalg import pinv
from sklearn.preprocessing import MinMaxScaler

def getDataset(id):
    n_components_range = []
    X = None
    y = None
    label = ''
    range_n_clusters = []
    if id == 1:
        print("Reading credit card data...")
        data = pd.read_csv('cctrain.csv')
        label = 'Credit Card'
        n_components_range = range(1,data.shape[1])
        range_n_clusters = range(2, 11)
    if id == 2:
        print("Reading sign language data...")
        data = pd.read_csv('sltrain.csv')
        label = 'Sign Language'
        frange = np.arange(1,data.shape[1])
        n_components_range = frange[np.mod(frange, 56) == 0].tolist()
        range_n_clusters = np.linspace(5,30,10, dtype='int').tolist()

    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    return X, y, label, n_components_range, range_n_clusters

def getReducedX(id, method):
    X = None
    label = ''
    if id == 1:
        label = 'Credit Card'
        filename = "%s-%s-Xt.csv" % (label.replace(" ", "-"), method)
        print("Reading %s reduced %s data..." % (method.lower(), label.lower()))
        X = pd.read_csv(filename)
    if id == 2:
        label = 'Sign Language'
        filename = "%s-%s-Xt.csv" % (label.replace(" ", "-"), method)
        print("Reading %s reduced %s data..." % (method.lower(), label.lower()))
        X = pd.read_csv(filename)

    return X

def saveXt(label, method, Xt, colprefix):
    ncolumns = Xt.shape[1]
    columns = map(lambda x: "%s%d" % (colprefix, x), np.arange(ncolumns)+1)
    filename = '%s-%s-Xt.csv' % (label.replace(" ", "-"), method)
    with open(filename, 'wb') as xf:
        xf.write(','.join(columns)+'\n')
        np.savetxt(xf, Xt, delimiter=',', fmt='%.10f' )

def reconstruct(component,X):
    if sps.issparse(component):
        W = component.todense()
    p = pinv(W)
    print("p", p.shape)
    print("W", W.shape)
    print("X", X.shape)
    r = np.dot(np.dot(p,W),(X.T)).T # Unproject projected data
    return r

def reconstructit(component, X):
    ax = np.dot(X, component.todense())
    scaler = MinMaxScaler()
    return scaler.fit_transform(ax)

def plot_clusters(label, method, methodname, X, y, nclasses, components):
    c0 = components[0]
    c1 = components[1]
    c2 = components[2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for j in range(nclasses):
        x = X.iloc[y == j,:]
        if x.shape[0] > 0:
            ax.scatter(x.iloc[:,c0], x.iloc[:,c1],  x.iloc[:,c2], marker='^')
    ax.set_xlabel(X.columns[c0])
    ax.set_ylabel(X.columns[c1])
    ax.set_zlabel(X.columns[c2])
    plt.title("%s %s Clusters (k=%d)" % (label, methodname, nclasses))
    plt.gcf()
    plt.savefig("%s-%s-clusters.png" % (label.replace(" ", "-"), method))
    plt.close()

def plot_scree(label, method, ver, n_components=None):
    print("plot scree...")
    n = n_components
    plt.plot(range(ver.shape[0]), ver, '.-')
    plt.plot(range(ver.shape[0]), np.cumsum(ver), '.-')
    plt.suptitle("%s %s Scree Plot" % (label, method))
    plt.title("n_components = %s" % ("None" if n_components is None else str(n_components)))
    plt.ylabel("Proportion of Variance Explained")
    plt.xlabel("Principal Component")
    plt.legend(["Variance", "Cumulative Variance"], loc="best")
    plt.gcf()
    filename = '%s-%s-%d-scree.png' % (label.replace(" ", "-"), method, n)
    plt.savefig(filename)
    plt.close()

def plot_re(label, method, mse, nc):
    print("plot re...")
    plt.plot(nc, mse, '.-')
    plt.ylabel("Reconstruction Error")
    plt.xlabel('n components')
    plt.title('%s %s Reconstruction Error' % (label, method))
    plt.gcf()
    plt.savefig(('%s %s reconstructionerror.png' % (label, method)).replace(" ", "-"), bbox_inches='tight')
    plt.close()

# reference:
# https://stackoverflow.com/questions/13224362/principal-component-analysis-pca-in-python
def biplot(label, method, score, coeff, labels):
    print("biplot...")
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("%s %s Biplot" % (label, method))
    plt.grid()
    plt.gcf()
    plt.savefig("%s-%s-biplot.png" % (label.replace(" ", "-"), method))
    plt.close()

def plot_first_images(firstimages, nc, method, label):
    print("plot %s reconstructed images..." % (method))
    fig=plt.figure(figsize=(8,8))
    columns = 4
    rows = 4
    for i in range(1, len(firstimages)+1):
        img = np.array(firstimages[i-1]).reshape((28,28))
        ax = fig.add_subplot(rows, columns, i)
        ax.title.set_text("Original" if i-1 == 0 else "nc=%d" % (nc[i-2]))
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img, cmap='gray')
    plt.suptitle("Original vs Reconstructed Images (%s)" % (method))    
    plt.gcf()
    plt.savefig(('%s %s reconstucted.png' % (label, method)).replace(" ", "-"))
    plt.close()

def plot_pdiff(label, method, pdiffms, pdiffstds, n_components):
    print("plot pdiff...")
    plt.plot(n_components, pdiffms, '.-')
    plt.fill_between(n_components, pdiffms - pdiffstds,
                     pdiffms + pdiffstds, alpha=0.1)
    plt.xlabel("n_components")
    plt.ylabel("% difference pairwise distances")
    plt.title("%s %s Pairwise Distance Differences" % (label, method))
    plt.gcf()
    plt.savefig("%s-%s-pdiff.png" % (label.replace(" ", "-"), method))
    plt.close()

def plot_silhscores(label, plotx, ploty, clustermethods):
    plt.plot(plotx[0], ploty[0], '.-')
    plt.plot(plotx[1], ploty[1], '.-')
    plt.ylabel("Silhouette Score")
    plt.xlabel("n_clusters")
    plt.title("%s Data: Silhouette Score" % (label.replace("-", " ")))
    plt.legend(clustermethods, loc="best")
    plt.gcf()
    plt.savefig("%s-score.png" % (label.replace(" ", "-")))
    plt.close()

def plot_basic_bar(xticks, yvalues, xlabel, ylabel, title, figname):
    plt.bar(xticks, yvalues)
    if len(xticks) > 20:
        nxt = np.array(xticks)
        nxt = nxt[np.mod(nxt, len(xticks)/20) == 1]
        xticks = nxt.tolist()
    plt.xticks(xticks)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.gcf()
    plt.savefig(figname)
    plt.close()

def plot_silh(label, method, name, n_clusters, X, cluster_labels, clusterer, silhouette_avg, sample_silhouette_values):
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

    if method == 'KM':
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
                  "with n_clusters = %d\nSilhoutte average %.3f" % (name, label.replace("-", " "), n_clusters, silhouette_avg)),
                  fontsize=14, fontweight='bold')
    plt.gcf()
    plt.savefig(label.replace(" ", "-")+'-'+method+'-'+str(n_clusters)+'.png')
    plt.close()

def plot_2bar(xdata1, xdata2, legends, xlabels, ylim, ylabel, title, figname):
        #data1 = {'cat1': [vals1], 'cat2': [vals1]}
        #data2 = {'cat1': [vals2], 'cat2': [vals2]}
        #legends = [vals1name, vals2name]
        width=0.8
        vals = [xdata1, xdata2]

        n = len(vals)
        _X = np.arange(len(xlabels))
        for i in range(n):
            plt.bar(_X - width/2. + i/float(n)*width, vals[i], 
                        width=width/float(n), align="edge")   
            plt.xticks(_X, xlabels)
        plt.ylim(ylim[0], ylim[1])
        plt.ylabel(ylabel)
        plt.legend(legends)
        plt.title(title)
        plt.gcf()
        plt.savefig(figname)
        plt.close()

def plot_2axis(y1, y2, x, ylabel1, ylabel2, xlabel, title, figname):
    fig, ax1 = plt.subplots()
    pts1 = ax1.plot(x, y1, 'o-', label=ylabel1)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1)
    
    ax2 = ax1.twinx()
    pts2 = ax2.plot(x, y2, '.-', label=ylabel2, linestyle='dashed', color="green", alpha=0.5)
    pts = pts1 + pts2
    labs = [p.get_label() for p in pts]
    ax2.set_ylabel(ylabel2, color="green")
    ax2.tick_params(axis='y', labelcolor='green')

    plt.title(title)
    plt.legend(pts, labs, loc='best')
    
    fig.tight_layout()
    fig = plt.gcf()
    fig.savefig(figname, bbox_inches="tight")
    plt.close()
