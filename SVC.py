# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 16:57:45 2016

@author: centraltendency
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

df = pd.read_csv("~/Santander/train.csv")

# Find columns == 0

zero_columns = df.columns[(df == 0).all() == True]
non_zero_columns = df.columns[(df == 0).all() == False]
new_df = df.drop(zero_columns, 1)
# cor_matrix = new_df.corr()

    

sc = StandardScaler()
x, y = new_df.iloc[:, 1:336].values, new_df.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
#x_train_std = sc.fit_transform(x_train)
#x_test_std = sc.fit_transform(x_test)

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV

pipe_svc = Pipeline([('scl', StandardScaler()),
                      ('clf', SVC(random_state=1))])
#scores_svc = cross_val_score(estimator=pipe_svc, X=x_train, y=y_train, cv=10, n_jobs=-1)
#print(scores_svc)

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'clf__C': param_range, 'clf__kernel': ['linear']},
              {'clf__C': param_range, 'clf__gamma': param_range, 'clf__kernel': ['rbf']}]
gs = GridSearchCV(estimator = pipe_svc,
                  param_grid = param_grid,
                  scoring = 'accuracy',
                  cv = 10,
                  n_jobs = -1)
gs1 = gs.fit(x_train, y_train)
print(gs1.best_score_)
print(gs1.best_params_)

