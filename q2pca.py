from __future__ import print_function

from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import numpy as np
from common import *

seed=42
method='PCA'

def applyPCA(label, method, X, n_components, reconstructimages=False, usen=None, seed=seed):
    print("doing %s..." % (method))
    mse = []
    firstimages = []
    model = None
    n = -1
    Xt = None

    for n in n_components:
      model = PCA(n_components=n, random_state=seed)
      Xt = model.fit_transform(X)
      Xr = model.inverse_transform(Xt)
      mse.append(mean_squared_error(X, Xr))
      firstimages.append(Xr[0,:])

    print("done. plotting...")

    plot_re(label, method, mse, n_components)
    if reconstructimages:
      firstimages.insert(0, np.array(X.iloc[0,:]))
      plot_first_images(firstimages, n_components, method, label)

    ver = model.explained_variance_ratio_
    evr = np.cumsum(model.explained_variance_ratio_)
    crange = np.where(evr > 0.95)   
    if usen is None:
      usen = X.shape[1]
      if len(crange[0]) > 0:
          usen = min(crange[0])+1
          print ("minimum # of components with variance explained > 0.95: %d" % (usen))
    biplot(label, method, Xt[:,0:2],np.transpose(model.components_[0:2, :]), X.columns.tolist())
    plot_scree(label, method, ver, n)
    
    print("%s: reducing components to %d..." % (method, usen))
    model = PCA(n_components=usen, random_state=seed)
    model = model.fit(X)
    Xt = model.transform(X)
    xticks = range(1,usen+1)
    yvalues = model.explained_variance_
    xlabel = 'principal components'
    ylabel = 'eigenvalues'
    title = '%s PCA Eigenvalues' % (label)
    figname = "%s-pca-eigenvalues.png" % (label.replace(" ", "-"))
    plot_basic_bar(xticks, yvalues, xlabel, ylabel, title, figname)

    return Xt, model

def reduceDim(X, model):
    return model.transform(X)

reconstructimages = False
usen = [18, 111]
for i in [1, 2]:
  print("="*10)
  model = None
  Xt = None
  for istest in [False, True]:
    X, y, label, n_components_range, _ = getDataset(i, istest)
    if not istest:
      Xt, model = applyPCA(label, method, X, n_components_range, reconstructimages, usen[i-1])
    else:
      Xt = reduceDim(X, model)    
    saveXt(label, method, Xt, "PC", istest)
    reconstructimages = False
  reconstructimages = True
  print("done.")