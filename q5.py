def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import timeit
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import strftime, localtime
from common import *

pstart = timeit.default_timer()
print("=========")
print('Start: {}'.format(strftime('%c', localtime())))
print("=========")

drmethods = ['PCA', 'ICA', 'RP', 'LDA']
clustermethods = ['KM', 'GM']
traintest = [['train', False], ['test', True]]
drnclusters = {
    'PCA': {'KM': 3, 'GM': 3},
    'ICA': {'KM': 9, 'GM': 2},
    'RP':  {'KM': 3, 'GM': 2},
    'LDA': {'KM': 2, 'GM': 2}
    }

# cluster test dataset
print("Clustering the test set")
print("-"*10)
seed = 42
for drmethod in drmethods:
    for id in [1,2]:
        Xtrain, ytrain, label, _ = getReducedX(id, drmethod, istest=False)
        Xtest, ytest, label, _ = getReducedX(id, drmethod, istest=True)
        for clustermethod in clustermethods:
            clusterer = None
            if clustermethod == 'GM':
                clusterer = GaussianMixture(n_components=drnclusters[drmethod][clustermethod], random_state=seed)        
            if clustermethod == 'KM':
                clusterer = KMeans(n_clusters=drnclusters[drmethod][clustermethod], random_state=seed)
            clusterer = clusterer.fit(Xtrain)
            cluster_labels = clusterer.predict(Xtrain)
            print("Saving %s %s %s cluster labels to csv..." % (label, drmethod, clustermethod))
            np.savetxt("%s-%s-%s-cluster-labels.csv" % (label.replace(" ", "-"), clustermethod, drmethod), cluster_labels, fmt="%d")
            cluster_labels = clusterer.predict(Xtest)
            print("Saving %s %s %s test cluster labels to csv..." % (label, drmethod, clustermethod))
            np.savetxt("%s-%s-%s-test-cluster-labels.csv" % (label.replace(" ", "-"), clustermethod, drmethod), cluster_labels, fmt="%d")
print('done.')
print
print("Modeling and predicting using reduced and clustered data")
print("-"*10)
seed = 0
for id in [1, 2]:
    model = None
    scorer = None
    traintime = {}
    trainscores = {}
    testscores = {}
    for clustermethod in clustermethods:
        trainscores[clustermethod] = {}
        testscores[clustermethod] = {}

    if id == 1:
        model = MLPClassifier(max_iter=1000, hidden_layer_sizes=100, alpha=0.1, activation='relu', solver='lbfgs', learning_rate='constant', random_state=seed)
        scorer = f1_score
        scorer_name = 'f1 score'
    if id == 2:
        model = MLPClassifier(max_iter=1000, hidden_layer_sizes=100, alpha=0.1, activation='tanh', random_state=seed)
        scorer = accuracy_score
        scorer_name = 'accuracy'

    for clustermethod in clustermethods:
        for drmethod in drmethods:
            columns = []
            enc = None
            for name, istest in traintest:
                Xrc, y, label, enc, columns = getReducedXwithEncodedLabels(id, drmethod, clustermethod, istest, columns, enc)
                if not istest:
                    print("Training...")
                    start = timeit.default_timer()
                    model.fit(Xrc, y)
                    traintime[drmethod] = timeit.default_timer()-start

                print("Predicting...")
                start = timeit.default_timer()
                ypred = model.predict(Xrc)
                score = scorer(y, ypred)
                print("%s + %s on %s %s Data %s: %.3f" % (drmethod, clustermethod, label, titlecase(name), scorer_name, score))
                if istest:
                    testscores[clustermethod][drmethod] = score
                else:
                    trainscores[clustermethod][drmethod] = score
            print("-"*5)
        
    figname = ("%s-%s-q5-train.png" % (label, scorer_name)).replace(" ", "-")
    plot_2bar(trainscores[clustermethods[0]].values(), 
        trainscores[clustermethods[1]].values(), 
        clustermethods, drmethods, [0., 1.], scorer_name, 
        '%s Train %s\n(Reduced and Clustered)' % (label, titlecase(scorer_name)), figname)
    figname = ("%s-%s-q5-test.png" % (label, scorer_name)).replace(" ", "-")
    plot_2bar(testscores[clustermethods[0]].values(), 
        testscores[clustermethods[1]].values(), 
        clustermethods, drmethods, [0., 1.], scorer_name, 
        '%s Test %s\n(Reduced and Clustered)' % (label, titlecase(scorer_name)), figname)

print
print("==========")
print('End: {}'.format(strftime('%c', localtime())))
print("==========")
print('Total Time: {} secs'.format(int(timeit.default_timer()-pstart)))
