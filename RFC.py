# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 00:01:44 2016

@author: centraltendency
"""

import pandas as pd
from sklearn.cross_validation import train_test_split

df = pd.read_csv("~/Santander/train.csv")

# Find columns == 0

zero_columns = df.columns[(df == 0).all() == True]
non_zero_columns = df.columns[(df == 0).all() == False]
new_df = df.drop(zero_columns, 1)
# cor_matrix = new_df.corr()

x, y = new_df.iloc[:, 1:336].values, new_df.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Decision Tree

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
clf = tree.fit(x_train, y_train)
clf.score(x_test, y_test)
clf.score(x_train, y_train)

from sklearn.cross_validation import cross_val_score
scores = cross_val_score(clf, x_train, y_train, cv=10, n_jobs=-1)
print(scores)

from sklearn.tree import export_graphviz
export_graphviz(tree, feature_names = df.columns, out_file = 'santander_tree.dot')

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(random_state=0)
clf_forest = forest.fit(x_train, y_train)
clf_forest.score(x_train, y_train)
clf_forest.score(x_test, y_test)

