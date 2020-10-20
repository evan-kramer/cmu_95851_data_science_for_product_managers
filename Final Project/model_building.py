# -*- coding: utf-8 -*-
'''
Evan Kramer
evankram
10/23/2020
'''
# Import modules
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import tree, ensemble
from sklearn.utils import resample
from sklearn.cluster import KMeans
from subprocess import call

# Load data, keep only the variables of interest and prep for cross-validation
folds = 5
np.random.seed(123)
dds11_clean = pd.read_csv('dds11_clean.csv')
dds11_clean = dds11_clean.astype('category')

# Keep variables of interest and create upsampled dataset
Y = dds11_clean[['plan_to_purchase_smartwatch', 'plan_to_purchase_fitness_band']]
X = (dds11_clean.filter(regex = '_cat|gender|income|region|employment|race_eth|children$|^own|plan_to_purchase|everyday_app_use|subscription_stream|subscription_news|news_vehicle')
                   .drop(columns = Y.columns)
                   .assign(fold = np.random.randint(0, folds, size = len(dds11_clean)))) 
# Try upsampling?
X_up_sw = (pd.concat([dds11_clean[dds11_clean.plan_to_purchase_smartwatch == 'No'],
                      resample(dds11_clean[dds11_clean.plan_to_purchase_smartwatch == 'Yes'],
                               replace = True,
                               n_samples = len(dds11_clean[dds11_clean.plan_to_purchase_smartwatch == 'No']),
                               random_state = 123)])[list(X.columns[X.columns != 'fold'])])
X_up_sw = X_up_sw.assign(fold = np.random.randint(0, folds, 
                                                  size = len(X_up_sw)))
X_up_fb = (pd.concat([dds11_clean[dds11_clean.plan_to_purchase_fitness_band == 'No'],
                      resample(dds11_clean[dds11_clean.plan_to_purchase_fitness_band == 'Yes'],
                               replace = True,
                               n_samples = len(dds11_clean[dds11_clean.plan_to_purchase_fitness_band == 'No']),
                               random_state = 123)])[list(X.columns[X.columns != 'fold'])])
X_up_fb = X_up_fb.assign(fold = np.random.randint(0, folds, 
                                                  size = len(X_up_fb)))
# Use categorical codes (to get numeric)
for v in X.columns:
    try:
        X[v] = X[v].cat.codes
        X_up_sw[v] = X_up_sw[v].cat.codes
        X_up_fb[v] = X_up_fb[v].cat.codes
    except:
        pass

# What are the market segments for “wearables”?
# Who is planning to purchase? 
# Cross-validate, train/test split, fit model
# X_train = X[X.fold > 2]
# X_test = X[X.fold <= 2]
# X_train = X[X.fold > 2].filter(regex = 'everyday_app')
# X_test = X[X.fold <= 2].filter(regex = 'everyday_app')
X_train = X_up_sw[X_up_sw.fold > 2].filter(regex = 'everyday_app')    
X_test = X_up_sw[X_up_sw.fold <= 2].filter(regex = 'everyday_app')    
# X_train = X_up_fb[X_up_fb.fold > 2].filter(regex = 'news')
# X_test = X_up_fb[X_up_fb.fold <= 2].filter(regex = 'news')        
Y_train = Y.loc[X_train.index]
Y_test = Y.loc[X_test.index] 
clf = ensemble.RandomForestClassifier()
clf.fit(X_train, Y_train.plan_to_purchase_smartwatch)
p = pd.DataFrame({'actual': Y_test.plan_to_purchase_smartwatch, 
                  'predicted': clf.predict(X_test)})

# Accuracy, precision, and recall
tp = len(p[(p.actual == 'Yes') & (p.predicted == 'Yes')])
fp = len(p[(p.actual == 'No') & (p.predicted == 'Yes')])
tn = len(p[(p.actual == 'No') & (p.predicted == 'No')])
fn = len(p[(p.actual == 'Yes') & (p.predicted == 'No')])
print('Accuracy: {accuracy:.1f}%'.format(accuracy = len(p[p.actual == p.predicted]) / len(p) * 100))
print('Precision: {precision:.1f}%'.format(precision = tp / (tp + fp) * 100))
print('Recall: {recall:.1f}%'.format(recall = tp / (tp + fn) * 100))

# Export tree
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi = 800)
tree.plot_tree(clf.estimators_[0],
               feature_names = X_test.columns, 
               class_names = 'own_smartwatch',
               filled = True)
fig.savefig('Visualizations/tree.png')

Figure out how to plot the decision tree with fewer estimators


# Who owns/uses wearables currently?
# Which advertising channels are more likely to get customers to buy? 
# Which apps do they use?
# Where do they get news? 



# age, gender, region, employment_status, race_ethnicity, children
# income_cat, own_*, plan_to_purchase_*, everyday_app_use_*, subscription_stream*, 
# subscription_news*, most_frequent_news_vehicle


# Model to categorize who currently owns smartwatches and fitness bands




# K-means clustering
# Set up k-means clustering algorithm 
lower_limit = 1
upper_limit = 11
sse = {}
for k in range(lower_limit, upper_limit):
    k_means = KMeans(n_clusters = k, init = 'k-means++', n_init = 10, max_iter = 300, 
                     random_state = 123)
    k_means.fit(X_up_fb)
    sse[k] = k_means.inertia_    
# Plot to see elbow
plt.plot(range(lower_limit, upper_limit), sse.values())
plt.xlabel('Clusters')
plt.ylabel('Sum of Squared Distances')
plt.xticks(range(lower_limit, upper_limit, 1))
plt.title('SSE as a Function of Number of Clusters')
plt.show()
# See features of cluster
k_means = KMeans(n_clusters = 5, init = 'k-means++', n_init = 10, max_iter = 300, 
                     random_state = 123)
k_means.fit(X_up_fb).labels_





# Model evaluation
# Precision vs. recall
# Area under receiver operating characteristic curve
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

# Partial dependence plots