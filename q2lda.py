from __future__ import print_function

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from common import *

seed=42
method='LDA'

def reduceDim(label, method, X, y, n_components, scorer, seed=seed):
    print("doing %s..." % (method))

    model = LinearDiscriminantAnalysis()
    Xt = model.fit_transform(X, y)
    ypred = model.predict(X)

    print("done. plotting...")

    n = Xt.shape[1]
    ver = model.explained_variance_ratio_
    print("%s: reduced components to %d..." % (method, n))
    print("total explained variance from reduced dimensions = %.3f" % (np.sum(ver)))
    if n > 1:
      plot_scree(label, method, ver, n)

    if scorer == "accuracy":
      print("accuracy score = %.3f" % (accuracy_score(y, ypred)))
    if scorer == "f1":
      print("f1 score = %.3f" % (f1_score(y, ypred)))
    
    return Xt

scorer = "f1"
for i in [1, 2]:
  print("="*10)
  X, y, label, n_components_range = getDataset(i)
  Xt = reduceDim(label, method, X, y, n_components_range, scorer)
  saveXt(label, method, Xt)
  scorer = "accuracy"
  print("done.")
