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
from sklearn import tree, ensemble, svm
from sklearn.utils import resample
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

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

# Use categorical codes (and crosswalk back)
categories = {}
for v in X.columns:
    try:
        categories[v] = {X[v].cat.codes.value_counts().index[i]: X[v].value_counts().index[i] for i in range(len(X[v].value_counts().index))}
        X[v] = X[v].cat.codes
        X_up_sw[v] = X_up_sw[v].cat.codes
        X_up_fb[v] = X_up_fb[v].cat.codes        
    except:
        pass

# What are the market segments for “wearables”?
# Who is planning to purchase? 
# Cross-validate, train/test split
X_train, X_test, Y_train, Y_test = {}, {}, {}, {}
pattern = 'subscription_stream|vehicle'
pattern = 'everyday_app'
pattern = 'news'
pattern = 'gender|age_cat|region|employment|children|income|fold'


X_train['plan_to_purchase_smartwatch'] = (X_up_sw[X_up_sw.fold >= 2]
                                            .filter(regex = pattern)
                                          )
X_test['plan_to_purchase_smartwatch'] = (X_up_sw[X_up_sw.fold < 2]
                                            .filter(regex = pattern)
                                         )
X_train['plan_to_purchase_fitness_band'] = (X_up_fb[X_up_fb.fold >= 2]
                                            .filter(regex = pattern)
                                            )
X_test['plan_to_purchase_fitness_band'] = (X_up_fb[X_up_fb.fold < 2]
                                            .filter(regex = pattern)
                                           )

# Loop for both fitness band and smartwatch
for v in ['plan_to_purchase_smartwatch', 'plan_to_purchase_fitness_band']:
    # Create Y training and test sets
    Y_train[v] = Y.loc[X_train[v].index]
    Y_test[v] = Y.loc[X_test[v].index]    
    
    # Logistic regression
    '''
    clf = LogisticRegression(random_state = 0, max_iter = 400)
    clf.fit(X_train[v], Y_train[v][v])
    clf.predict(X_test[v])
    # Strongest predictors
    print(v.replace('_', ' ').title())
    for f in [True, False]:
        print((pd.DataFrame({'var': X_train[v].columns, 
                             'coef': clf.fit(X_train[v], Y_train[v][v]).coef_.reshape(len(X_train[v].columns),)})
               .sort_values('coef', ascending = f))
              .head(5))       
    '''
    
    # SVM
    '''
    clf = svm.SVC(kernel = 'linear') # could also use polynomial or RBF 
    clf.fit(X_train[v], Y_train[v][v])
    '''
    
    # Random forest
    max_depth = None
    clf = ensemble.RandomForestClassifier(max_depth = max_depth) # set max depth to interpret
    clf.fit(X_train[v], Y_train[v][v])
    
    # Plot feature importances; cue from this piece: https://explained.ai/rf-importance/
    fi = (pd.merge(pd.DataFrame({'feature': X_test[v].columns}),
                   (pd.DataFrame({'importance': clf.feature_importances_}, 
                                 index = np.argsort(clf.feature_importances_)[::-1])
                    .sort_index()),
                   how = 'left', left_index = True, right_index = True)
          .sort_values(by = 'importance')
          .tail(10)
          .reset_index()
          .drop(columns = 'index'))
    plt.figure(figsize = 1 * np.array([9.32, 3.74]))
    plt.title("Most Important Features " + 
              v.replace('plan_to_purchase_', '- ').replace('_', ' ').title())
    plt.barh(fi.feature, fi.importance, height = 0.7)
    plt.yticks(ticks = fi.sort_index().index,
                labels = (fi.feature
                        .str.replace('everyday_app_use_', '')
                        .str.replace('cat', '')
                        .str.replace('_', ' ').str.title()))
    plt.savefig('Visualizations/feature_importances_' + v + '.png')
    
    # Plot one tree from random forest
    plt.figure(figsize = 7 * np.array([6.4, 4.8]))
    tree.plot_tree(clf.fit(X_train[v], Y_train[v][v]).estimators_[0], 
                   feature_names = X_train[v].columns, filled = True)
    plt.savefig('Visualizations/tree_' + v + '.png')
        
    # Accuracy, precision, and recall
    print(v.replace('_', ' ').title())
    p = pd.DataFrame({'actual': Y_test[v][v], 
                      'predicted': clf.predict(X_test[v])})
    tp = len(p[(p.actual == 'Yes') & (p.predicted == 'Yes')])
    fp = len(p[(p.actual == 'No') & (p.predicted == 'Yes')])
    tn = len(p[(p.actual == 'No') & (p.predicted == 'No')])
    fn = len(p[(p.actual == 'Yes') & (p.predicted == 'No')])
    print('Accuracy: {accuracy:.1f}%'.format(accuracy = len(p[p.actual == p.predicted]) / len(p) * 100))
    print('Precision: {precision:.1f}%'.format(precision = tp / (tp + fp) * 100))
    print('Recall: {recall:.1f}%'.format(recall = tp / (tp + fn) * 100))
    
    # Plot AUC
    plt.figure(figsize = 1.5 * np.array([6.4, 4.8]))
    ns_probs = [0 for _ in range(len(Y_test[v][v]))]
    rf_probs = clf.fit(X_train[v], Y_train[v][v]).predict_proba(X_test[v])[:, 1] # keep only positive outcome
    ns_auc = roc_auc_score(Y_test[v][v], ns_probs)
    rf_auc = roc_auc_score(Y_test[v][v], rf_probs)
    ns_fpr, ns_tpr, _ = roc_curve(Y_test[v][v], ns_probs, pos_label = 'Yes')
    lr_fpr, lr_tpr, _ = roc_curve(Y_test[v][v], rf_probs, pos_label = 'Yes')
    plt.plot(ns_fpr, ns_tpr, linestyle = '--', 
             label = 'Null Model, AUC = {auc:.3f}'.format(auc = ns_auc))
    plt.plot(lr_fpr, lr_tpr, marker = '.', 
             label = 'Random Forest, AUC = {auc:.3f}'.format(auc = rf_auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve\n' + v.replace('plan_to_purchase_', '').replace('_', ' ').title())
    plt.legend()
    plt.savefig('Visualizations/aurocc_' + v + '.png')

    # Market segments and sizing
    plt.figure()
    dds11_clean.most_frequent_news_vehicle.hist(by = dds11_clean[v], sharex = True)
    plt.savefig('Visualizations/by_news_outlet_' + v)
    dds11_clean.subscription_stream_music.hist(by = dds11_clean[v], sharex = True)
    plt.figure()
    dds11_clean.income_cat.hist(by = dds11_clean[v], sharex = True, align = 'left')
    plt.savefig('Visualizations/by_income_' + v)
    dds11_clean.age_cat.hist(by = dds11_clean[v], sharex = True, align = 'left')
    dds11_clean.gender.hist(by = dds11_clean.plan_to_purchase_smartwatch, sharex = True)
    dds11_clean.everyday_app_use_fitness_health.hist(by = dds11_clean[v], sharex = True)
    
# K-means clustering and calculate group tendencies?
'''
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
'''

# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
