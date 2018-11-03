from __future__ import print_function

from sklearn.decomposition import FastICA
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from common import *

seed=42
method='ICA'

def applyICA(label, method, X, n_components, reconstructimages=False, seed=seed):
    print("doing %s..." % (method))
    mse = []
    firstimages = []
    model = None
    n = -1
    Xt = None
    ngratio = []
    meank = []

    for n in n_components:
      model = FastICA(n_components=n, random_state=seed)
      Xt = model.fit_transform(X)
      Xr = model.inverse_transform(Xt)
      mse.append(mean_squared_error(X, Xr))
      firstimages.append(Xr[0,:])
      k = pd.DataFrame(Xt).kurtosis().abs()
      meank.append(k.mean())
      ngratio.append(len(k[k>2])*1./len(k))

    print("done. plotting...")
    plot_re(label, method, mse, n_components)
    if reconstructimages:
      firstimages.insert(0, np.array(X.iloc[0,:]))
      plot_first_images(firstimages, n_components, method, label)

    print("mean kurtosis = %.5f" % (k.mean()))  
    print("total no of components = %d" % (len(k)))  
    print("no of components that are non-gaussian = %d" % (len(k[k>2.])))

    plot_2axis(meank, ngratio, n_components, 'mean kurtosis', 'ratio non-gaussian to gaussian', 'no of components', 
      '%s kurtosis and non-gaussian components' % (method), 
      '%s-%s-kurt-ng.png' % (label.replace(" ", "-"), method))

def reduceDim(method, X, n):
    print("%s: reducing components to %d..." % (method, n))
    model = FastICA(n_components=n, random_state=seed)
    Xt = model.fit_transform(X)
    return Xt

reconstructimages = False
usen = [15, 112]
for i in [1, 2]:
  print("="*10)
  X, y, label, n_components_range, range_n_clusters = getDataset(i)
  applyICA(label, method, X, n_components_range, reconstructimages)
  Xt = reduceDim(method, X, usen[i-1])
  saveXt(label, method, Xt, "IC")
  reconstructimages = True
  print("done.")