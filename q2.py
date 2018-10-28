from __future__ import print_function

from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from common import *

seed=42
method='PCA'

def reduceDim(label, method, X, n_components=[], seed=seed):
    print("doing %s..." % (method))
    mse = []
    firstimages = []
    pca = None
    n = -1
    for n in n_components:
        #print("n=%d" % (n))
        pca = PCA(n_components=n, random_state=seed)
        pca.fit(X)
        Xt = pca.transform(X)
        Xr = pca.inverse_transform(Xt)
        mse.append(mean_squared_error(X, Xr))
        firstimages.append(Xr[0,:])
    print("done. plotting...")
    ver = pca.explained_variance_ratio_
    print ("=== %s n_components = %d ===" % (method, n))
    evr = np.cumsum(pca.explained_variance_ratio_)
    crange = np.where(evr > 0.95)   
    print ("%s components with greater than 0.95 variance explained:" % (label))
    if len(crange[0]) > 0:
        print("Minimum # of components: %d" % (min(crange[0])+1))
        #print("Maximum # of components: %d" % (max(crange[0])+1))
    #print(X.columns[:2].tolist())
    biplot(label, method, Xt[:,0:2],np.transpose(pca.components_[0:2, :]), X.columns.tolist())
    plot_scree(label, method, ver, n)
    return mse, n_components, firstimages    

print("Reading credit card data...")
data = pd.read_csv('cctrain.csv')
X = data.iloc[:,:-1]

label = 'Credit Card'
nc = np.arange(X.shape[1])+1
mse, nc, firstimages = reduceDim(label, method, X, nc[1:])
plot_re(label, method, mse, nc)

print("Reading sign language data...")
sldata = pd.read_csv('sltrain.csv')
Xsl = sldata.iloc[:,:-1]

label = 'Sign Language'
frange = np.arange(Xsl.shape[1])+1
nc = frange[np.mod(frange, 56) == 0]
mse, nc, firstimages = reduceDim(label, method, Xsl, nc.tolist())
plot_re(label, method, mse, nc)

firstimages.insert(0, np.array(Xsl.iloc[0,:]))
plot_first_images(firstimages, nc, method, label)