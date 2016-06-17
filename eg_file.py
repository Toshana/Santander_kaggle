# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 18:49:27 2016

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
x, y = new_df.iloc[:, 1:].values, new_df.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
x_train_std = sc.fit_transform(x_train)
x_test_std = sc.fit_transform(x_test)


# accessing feature importance with random forests

import numpy as np
from sklearn.ensemble import RandomForestClassifier
feat_labels = new_df.columns[1:]
forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
forest.fit(x_train, y_train)
importances = forest.feature_importances_
indicies = np.argsort(importances)[::-1]
for f in range(x_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[f], importances[indicies[f]]))


# Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('pca', PCA(n_components=5)),
                     ('clf', LogisticRegression(penalty = 'l2', random_state=1))])
from sklearn.cross_validation import cross_val_score
scores_lr = cross_val_score(estimator=pipe_lr, X=x_train, y=y_train, cv=10, n_jobs=-1)
print(scores_lr)

## learning curve function

import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve

train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr, X=x_train, y=y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=-1)
train_mean = np.mean(train_scores, axis = 1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker = 's', markersize=5, label='validation accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.0])
plt.show()

# SVM

from sklearn.svm import SVC
pipe_svc = Pipeline([('scl', StandardScaler()),
                     ('pca', PCA(n_components=5)),
                      ('clf', SVC(kernel='rbf'))])
scores_svc = cross_val_score(estimator=pipe_svc, X=x_train, y=y_train, cv=10, n_jobs=-1)
print(scores_svc)

## learning curve function

import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve

train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_svc, X=x_train, y=y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=-1)
train_mean = np.mean(train_scores, axis = 1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker = 's', markersize=5, label='validation accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.0])
plt.show()



