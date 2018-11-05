from __future__ import print_function

from sklearn.random_projection import SparseRandomProjection
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances, mean_squared_error
import numpy as np
import pandas as pd
from common import *

method='RP'

def applyRP(label, method, X, n_components, usen, reconstructimages=False):
    print("doing %s..." % (method))
    pdiffms = []
    pdiffstds = []
    mse = []
    firstimages = []

    for n in n_components:
      model = SparseRandomProjection(n_components=n)
      Xt = model.fit_transform(X)
      Xr = reconstructit(model.components_, Xt)
      mse.append(mean_squared_error(X, Xr))
      firstimages.append(Xr[0,:])

      Xtd = pairwise_distances(Xt)
      Xd = pairwise_distances(X)

      nonzero = Xd != 0
      Xd = Xd[nonzero]

      pdiff = np.abs(Xtd[nonzero]-Xd)/Xd
      pdiffm = pdiff.mean()
      pdiffstd = pdiff.std()
      pdiffms.append(pdiffm)
      pdiffstds.append(pdiffstd)

    print("done. plotting...")

    plot_pdiff(label, method, np.array(pdiffms), np.array(pdiffstds), n_components)
    plot_re(label, method, mse, n_components)
    if reconstructimages:
      firstimages.insert(0, np.array(X.iloc[0,:]))
      plot_first_images(firstimages, n_components, method, label)

    model = SparseRandomProjection(n_components=usen)
    model = model.fit(X)
    return model

def reduceDim(method, X, n, model):
    print("%s: reducing components to %d..." % (method, n))
    Xt = model.transform(X)
    return Xt

def plot_jl_bounds(label, X):
    """
    http://scikit-learn.org/stable/auto_examples/plot_johnson_lindenstrauss_bound.html#sphx-glr-auto-examples-plot-johnson-lindenstrauss-bound-py
    """
    print("calculating jl bounds")
    eps_ranges = []
    eps_ranges.append(np.linspace(0.2, 0.99, 5))

    # range of number of samples (observation) to embed
    n_samples_range = np.linspace(100, 6000,5)

    for i, eps_range in enumerate(eps_ranges):
        colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(eps_range)))
        plt.figure()
        for eps, color in zip(eps_range, colors):
            min_n_components = johnson_lindenstrauss_min_dim(n_samples_range, eps=eps)
            plt.plot(n_samples_range, min_n_components, color=color)

        plt.legend(["eps = %0.1f" % eps for eps in eps_range], loc="best")
        plt.xlabel("Number of observations to eps-embed")
        plt.ylabel("Minimum number of dimensions")
        plt.title("Johnson-Lindenstrauss bounds:\n%s Data" % (label))
        plt.axhline(y=X.shape[1], color='r', linestyle='--', alpha=0.3)
        plt.axvline(x=X.shape[0], color='r', linestyle='--', alpha=0.3)
        plt.gcf()
        plt.savefig('%s-jlbounds.png' % (label.replace(" ","-")))
        plt.close()

reconstructimages = False
usen = [11, 280]
for i in [1, 2]:
  print("="*10)
  model = None
  X, _, label, n_components_range, _ = getDataset(i)
  for j in range(5):
    model = applyRP(label, method+str(j), X, n_components_range, usen[i-1], reconstructimages)
  for istest in[False, True]:
    if istest:
      X, _, label, _, _ = getDataset(i, istest)
    Xt = reduceDim(method, X, usen[i-1], model)
    saveXt(label, method, Xt, "RP", istest)
    if not istest:
      plot_jl_bounds(label, X)
  reconstructimages = True
  print("done.")
