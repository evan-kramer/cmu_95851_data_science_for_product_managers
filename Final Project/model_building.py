# -*- coding: utf-8 -*-
'''
Evan Kramer
evankram
10/23/2020
'''
# Import modules
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree, ensemble
from subprocess import call

# Load data, keep only the variables of interest and prep for cross-validation
folds = 5
dds11_clean = pd.read_csv('dds11_clean.csv')

# Model to categorize who currently owns smartwatches and fitness bands
# Check for balanced classes; up-sample and down-sample
# Decision trees (random forest)
x = pd.get_dummies(dds11_clean[['age', 'gender', 'region', 'children', 'income_cat']])#, 
                 # 'own_tv':'own_drone', 'subscription_cable':'subscription_magazine',
                 # 'everyday_app_use_photo_video':'everyday_app_use_hobbies']]
y = dds11_clean['own_smartwatch']
clf = ensemble.RandomForestClassifier()
clf.fit(x, y)
clf.predict(x)
clf.decision_path(x)
# clf.predict(X, y)
# K-means clustering
# Export tree

# Export as dot file
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi = 800)
tree.plot_tree(clf.estimators_[0],
               feature_names = x.columns, 
               class_names = 'own_smartwatch',
               filled = True)
fig.savefig('Visualizations/tree.png')


# Model evaluation
# Precision vs. recall
# Area under receiver operating characteristic curve
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html