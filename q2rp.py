from __future__ import print_function

from sklearn.random_projection import SparseRandomProjection
from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd
from common import *

seed=42
method='RP'

def applyRP(label, method, X, n_components, reconstructimages=False, seed=seed):
    print("doing %s..." % (method))
    pdiffms = []
    pdiffstds = []

    for n in n_components:
      model = SparseRandomProjection(n_components=n, random_state=seed)
      Xt = model.fit_transform(X)

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

def reduceDim(method, X, n):
    print("%s: reducing components to %d..." % (method, n))
    model = SparseRandomProjection(n_components=n, random_state=seed)
    Xt = model.fit_transform(X)
    return Xt

reconstructimages = False
usen = [11, 280]
for i in [1, 2]:
  print("="*10)
  X, y, label, n_components_range = getDataset(i)
  applyRP(label, method, X, n_components_range, reconstructimages)
  Xt = reduceDim(method, X, usen[i-1])
  saveXt(label, method, Xt)
  print("done.")
