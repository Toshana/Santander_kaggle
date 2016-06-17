# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 15:15:09 2016

@author: centraltendency
"""

from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

class SBS():
    def __init__(self, estimator, k_features, scoring=accuracy_score, test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state
        
    def fit(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_size, random_state=self.random_state)
        
        dim = x_train.shape[1]
        self.indicies_ = tuple(range(dim))
        self.subsets_ =[self.indicies_]
        score = self._calc_score(x_train, y_train, x_test, y_test, self.indicies_)
        self.scores_ = [score]
        
        while dim > self.k_features:
            scores = []
            subsets = []
            
            for p in combinations(self.indicies_, r=dim-1):
                score = self._calc_score(x_train, y_train, x_test, y_test, p)
                scores.append(score)
                subsets.append(p)
                
            best = np.argmax(scores)
            self.indicies_ = subsets[best]
            self.subsets_.append(self.indicies_)
            dim -= 1
            
            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]
        
        return self
        
    def transform(self, x):
        return x[:, self.indicies_]
        
    def _calc_score(self, x_train, y_train, x_test, y_test, indicies):
        self.estimator.fit(x_train[:, indicies], y_train)
        y_pred = self.estimator.predict(x_test[:, indicies])
        score = self.scoring(y_test, y_pred)
        return score
        
        
# import santander dataset

import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("~/Santander/train.csv")
zero_columns = df.columns[(df == 0).all() == True]
non_zero_columns = df.columns[(df == 0).all() == False]
new_df = df.drop(zero_columns, 1)
sc = StandardScaler()
x, y = new_df.iloc[:, 1:].values, new_df.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
x_train_std = sc.fit_transform(x_train)
x_test_std = sc.fit_transform(x_test)

# SBS implementation using KNN

from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
knn = KNeighborsClassifier(n_neighbors=2)
sbs = SBS(knn, k_features=1)
sbs.fit(x_train_std, y_train)

k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.1])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.show()