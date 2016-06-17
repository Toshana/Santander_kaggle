# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 11:45:40 2016

@author: centraltendency
"""

# Nested cross validation

from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
import pandas as pd

df = pd.read_csv("~/Santander/train.csv")
x, y = df.iloc[:, 1:].values, df.iloc[:, -1].values

# separate data into training and test sets

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)

from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score

pipe_svc = Pipeline([('scl', StandardScaler()), ('clf', SVC(random_state=1))])
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'clf__C': param_range, 
               'clf__kernel': ['linear']}, {'clf__C': param_range, 'clf__gamma': param_range, 'clf__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)
scores = cross_val_score(gs, x, y, scoring='accuracy', cv=5)
print('CV accuracy: %.3f +/-%.3f' % (np.mean(scores), np.std(scores)))

