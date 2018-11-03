def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import timeit
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from copy import copy
import json
from time import strftime, localtime
from common import *

pstart = timeit.default_timer()
print("=========")
print('Start: {}'.format(strftime('%c', localtime())))
print("=========")

seed = 0
drmethods = ['PCA', 'ICA', 'RP', 'LDA']
traintest = [['train', False], ['test', True]]

for id in [1, 2]:
    model = None
    scorer = None
    X = {}
    y = {}
    traintime = {}
    scores = {}
    for name, istest in traintest:
        scores[name] = {}
    for drmethod in drmethods:
        if id == 1:
            model = MLPClassifier(max_iter=1000, hidden_layer_sizes=100, alpha=0.1, activation='relu', solver='lbfgs', learning_rate='constant', random_state=seed)
            scorer = f1_score
            scorer_name = 'f1 score'
        if id == 2:
            model = MLPClassifier(max_iter=1000, hidden_layer_sizes=100, alpha=0.1, activation='tanh', random_state=seed)
            scorer = accuracy_score
            scorer_name = 'accuracy'

        for name, istest in traintest:
            X[name], y[name], label, _ = getReducedX(id, drmethod, istest=istest)

        print("Training...")
        start = timeit.default_timer()
        model.fit(X['train'], y['train'])
        traintime[drmethod] = timeit.default_timer()-start

        print("Predicting...")
        for name, istest in traintest:
            start = timeit.default_timer()
            ypred = model.predict(X[name])
            #predtime[name][drmethod] = timeit.default_timer()-start
            scores[name][drmethod] = scorer(y[name], ypred)
            print("%s %s %s: %.3f" % (drmethod, name, scorer_name, scores[name][drmethod]))
  
    figname = ("%s-%s.png" % (label, scorer_name)).replace(" ", "-")
    plot_2bar(scores['train'].values(), scores['test'].values(), ['train', 'test'], drmethods, [0., 1.], scorer_name, '%s Train vs Test %s' % (label, titlecase(scorer_name)), figname)

print
print("==========")
print('End: {}'.format(strftime('%c', localtime())))
print("==========")
print('Total Time: {} secs'.format(int(timeit.default_timer()-pstart)))
