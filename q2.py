from __future__ import print_function

from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import SparseRandomProjection, johnson_lindenstrauss_min_dim
from sklearn.metrics import mean_squared_error, pairwise_distances
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from common import *
import sys

seed=42
def Usage():
  print("Usage: %s <PCA|ICA|RP>" % (sys.argv[0]))

if len(sys.argv) < 2:
  Usage()
  sys.exit(1)

method=sys.argv[1]

def reduceDim(label, method, X, n_components=[], seed=seed):
    print("doing %s..." % (method))
    mse = []
    firstimages = []
    model = None
    n = -1
    Xt = None
    pdiffms = []
    pdiffstds = []

    for n in n_components:
      if method == 'PCA':
        model = PCA(n_components=n, random_state=seed)
      if method == 'ICA':
        model = FastICA(n_components=n, random_state=seed)
      if method == 'RP':
        model = SparseRandomProjection(n_components=n, random_state=seed)

      Xt = model.fit_transform(X)

      if method == 'RP':
        Xtd = pairwise_distances(Xt)
        Xd = pairwise_distances(X)
        nonzero = Xd != 0
        Xd = Xd[nonzero]
        pdiff = np.abs(Xtd[nonzero]-Xd)/Xd
        pdiffm = pdiff.mean()
        pdiffstd = pdiff.std()
        pdiffms.append(pdiffm)
        pdiffstds.append(pdiffstd)
      if method in ['PCA', 'ICA']:   
        Xr = model.inverse_transform(Xt)
        mse.append(mean_squared_error(X, Xr))
        firstimages.append(Xr[0,:])

    print("done. plotting...")

    if method in ['PCA', 'ICA']:
      plot_re(label, method, mse, n_components)
      firstimages.insert(0, np.array(Xsl.iloc[0,:]))
      plot_first_images(firstimages, n_components, method, label)

    if method == 'PCA':
      ver = model.explained_variance_ratio_
      print ("=== %s n_components = %d ===" % (method, n))
      evr = np.cumsum(model.explained_variance_ratio_)
      crange = np.where(evr > 0.95)   
      usen = X.shape[1]
      if len(crange[0]) > 0:
          usen = min(crange[0])+1
          print ("minimum # of components with variance explained > 0.95: %d" % (usen))
      biplot(label, method, Xt[:,0:2],np.transpose(model.components_[0:2, :]), X.columns.tolist())
      plot_scree(label, method, ver, n)
      model = PCA(n_components=usen, random_state=seed)
      Xt = model.fit_transform(X)

    if method == 'ICA':
      k = pd.DataFrame(Xt).kurtosis().abs()
      print("mean kurtosis = %.5f" % (k.mean()))  
      print("total no of components = %d" % (len(k)))  
      print("no of components that are non-gaussian = %d" % (len(k[k>2.])))
    if method == 'RP':
      plot_pdiff(label, method, np.array(pdiffms), np.array(pdiffstds), n_components)

    np.savetxt('%s-%s-Xt.csv' % (label, method), Xt, delimiter=',', fmt='%.10f' )
    return mse, n_components, firstimages    

print("Reading credit card data...")
data = pd.read_csv('cctrain.csv')
X = data.iloc[:,:-1]

label = 'Credit Card'
nc = np.arange(X.shape[1])+1
mse, nc, firstimages = reduceDim(label, method, X, nc[1:])

print("Reading sign language data...")
sldata = pd.read_csv('sltrain.csv')
Xsl = sldata.iloc[:,:-1]

label = 'Sign Language'
frange = np.arange(Xsl.shape[1])+1
nc = frange[np.mod(frange, 56) == 0]
mse, nc, firstimages = reduceDim(label, method, Xsl, nc.tolist(), True)