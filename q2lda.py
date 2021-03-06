from __future__ import print_function

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from common import *

seed=42
method='LDA'

def applyLDA(label, method, X, y, scorer, istest=False, model=None, seed=seed):
    print("doing %s..." % (method))
    score = None

    if not istest:
      model = LinearDiscriminantAnalysis()
      model = model.fit(X, y)
    Xt = model.transform(X)
    ypred = model.predict(X)

    print("done. plotting...")

    n = Xt.shape[1]
    ver = model.explained_variance_ratio_
    print("%s: reduced components to %d..." % (method, n))
    print("total explained variance from reduced dimensions = %.3f" % (np.sum(ver)))
    if n > 1:
      plot_scree(label, method, ver, n)

    if scorer == "accuracy":
      score = accuracy_score(y, ypred)
      print("accuracy score = %.3f" % (score))
    if scorer == "f1":
      score = f1_score(y, ypred)
      print("f1 score = %.3f" % (score))
    
    return Xt, score, model

def plot_scores(scores, labels, istest):
    figname = "%s-score.png" % (method)
    if istest:
      figname = "%s-test-score.png" % (method)
    xticks = range(2)
    plt.bar(xticks, scores)
    plt.text(-0.2, scores[0]/2., 'f1 score = %.3f' % (scores[0]), color="white")
    plt.text(0.8, scores[1]/2., 'accuracy = %.3f' % (scores[1]), color="white")
    plt.ylabel('f1 / accuracy score')
    plt.xticks(xticks, labels)
    plt.title("LDA Performance")
    if istest:
      plt.title("LDA Performance (Test)")
    plt.gcf()
    plt.savefig(figname, bbox_inches = "tight")
    plt.close()

models = []
for istest in [False, True]:
  scorer = "f1"
  scores = []
  labels = []
  for i in [1, 2]:
    print("="*10)
    X, y, label, _, _ = getDataset(i, istest)
    if not istest:
      Xt, score, model = applyLDA(label, method, X, y, scorer, istest, None)
      models.append(model)
    else:
      Xt, score, model = applyLDA(label, method, X, y, scorer, istest, models[i-1])
    saveXt(label, method, Xt, "LD", istest)
    scores.append(score)
    labels.append(label)
    scorer = "accuracy"
  plot_scores(scores, labels, istest)
print("done.")
