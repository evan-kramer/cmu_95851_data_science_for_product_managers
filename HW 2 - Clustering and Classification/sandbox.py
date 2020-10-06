# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 21:46:30 2020

@author: evan.kramer
"""
# Prep
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('analytic_data2020_0.csv', header = 0, skiprows = 1)
data.head()
# data.describe()
# data.info()
data_prelim = (data.filter(regex = '(code|rawvalue|state|county|year)') # keep only variables that match these regexs
             .query('statecode != 0 and countycode != 0')) # keep only county-level amounts

# Investigate missing values
data_prelim.describe().transpose()
data_prelim.info()
{np.dtype(data_prelim[i]) for i in data_prelim.columns}
m = {}
for i in data_prelim.columns:
    if np.dtype(data_prelim[i]) in ['int64', 'float64']:
        m[i] = np.isnan(data_prelim[i]).sum()
    else:
        m[i] = np.sum(data_prelim[i].isna() == True)

# Join with regional data to create regional average
regions = (pd.read_csv('states.csv').filter(['State Code', 'Region', 'Division']))
by_region = pd.merge(data_prelim, regions, how = 'left', left_on = 'state', right_on = 'State Code') 

# Plot missingness and examine distributions
'''
plt.hist(m.values(), bins = 30)
plt.title('Count of Missing Values across Variables')
plt.show()

for i in by_region.columns: 
    plt.hist(by_region[i], bins = 30)
    plt.title(i)
    plt.show()
    
for i in by_region.select_dtypes(['int64', 'float64']).columns:
    plt.boxplot(by_region[i])
    plt.show()
'''

# Impute missing values where possible
for i in by_region.columns:
    if np.dtype(by_region[i]) in ['int64', 'float64']:
        # State average
        by_region[i] = by_region[i].where(pd.notnull(by_region[i]), by_region[['state', i]].dropna().groupby('state')[i].mean())
        # Regional average
        by_region[i] = by_region[i].where(pd.notnull(by_region[i]), by_region[['Region', i]].dropna().groupby('Region')[i].mean())
        # National average
        by_region[i] = by_region[i].where(pd.notnull(by_region[i]), data.query('statecode == 0 & countycode == 0')[i]) 
        # Global mean
        by_region[i] = by_region[i].where(pd.notnull(by_region[i]), np.mean(by_region[i]))

# Examine correlations between remaining variables


by_region.corr()#[by_region.corr().abs() < 0.8]

# Check missing values again
m = {}
for i in data_prelim.columns:
    if np.dtype(data_prelim[i]) in ['int64', 'float64']:
        m[i] = np.isnan(data_prelim[i]).sum()
    else:
        m[i] = np.sum(data_prelim[i].isna() == True)
        
# Standardize data 
for i in by_region.loc[:, 'v001_rawvalue':'v097_rawvalue'].columns:
# for i in by_region.select_dtypes(['int64', 'float64']).columns:
    by_region[i] = (by_region[i] - np.mean(by_region[i])) / np.std(by_region[i])

# Drop all variables in which at least half of the records have missing values
by_region = by_region.loc[:, (by_region.isnull().sum(axis=0) <= len(by_region) / 2)]

# Check for and address outliers
ids = by_region.loc[:, 'statecode':'year']
var = by_region.iloc[:, np.r_[7:81]]
var = var[var < 3]
var = var[var > -3]
data_clean = pd.merge(ids, var, left_index = True, right_index = True)
data_clean

# Double-check missingness
m3 = {}
for i in data_clean.columns:
    if np.dtype(data_clean[i]) in ['int64', 'float64']:
        m3[i] = np.isnan(data_clean[i]).sum()
    else:
        m3[i] = np.sum(data_clean[i].isna() == True)
m3

# Fill with 0 (same as average for that column, since we standardized the data)
data_imputed = data_clean.fillna(0).iloc[:, np.r_[2, 6:42]]
# Drop outliers
data_complete = data_clean.dropna().iloc[:, np.r_[2, 6:42]] # only left with 2/3 of the records 

# Set up k-means clustering algorithm
k_means = KMeans(
    init = 'k-means++',
    n_clusters = 5,
    n_init = 10,
    max_iter = 300,
    random_state = 123
)

# Fit model (with both imputed and complete data)
k_means.fit(data_imputed)
# kmeans.fit(data_complete)
k_means.inertia_
k_means.cluster_centers_
k_means.n_iter_
k_means.labels_

lower_limit = 1
upper_limit = 11
sse_imputed = {}
sse_complete = {}
for k in range(lower_limit, upper_limit):
    k_means = KMeans(n_clusters = k, init = 'k-means++', n_init = 10, max_iter = 300, random_state = 123)
    k_means.fit(data_imputed)
    sse_imputed[k] = k_means.inertia_
    k_means.fit(data_complete)
    sse_complete[k] = k_means.inertia_
    
# Set up multiple plots
figure, (plot1, plot2) = plt.subplots(1, 2, sharex = 'all', sharey = 'all')
# figure.suptitle('Comparing Complete and Imputed Data')
# Plot 1
plot1.plot(range(lower_limit, upper_limit), sse_imputed.values())
plot1.set_xlabel('Clusters')
plot1.set_ylabel('Sum of Squared Distances')
plot1.set_xticks(range(lower_limit, upper_limit, 1))
plot1.set_title('Imputation with 0 (Average)')
# Plot 2
plot2.plot(range(lower_limit, upper_limit), sse_complete.values())
plot2.set_xlabel('Clusters')
plot2.set_xticks(range(lower_limit, upper_limit, 1))
plot2.set_title('Complete Cases, No Outliers')
plt.show()

# Optimal number of clusters looks like 4
# Display which counties are grouped together
k_means = KMeans(n_clusters = 4, init = 'k-means++', n_init = 10, max_iter = 300, random_state = 123)
k_means.fit(data_imputed)
data_imputed['cluster_label'] = k_means.labels_
k_means = KMeans(n_clusters = 4, init = 'k-means++', n_init = 10, max_iter = 300, random_state = 123)
k_means.fit(data_complete)
data_complete['cluster_label'] = k_means.labels_
# display(data_imputed)
# display(data_complete)

pd.merge(ids, data_complete[['fipscode', 'cluster_label']], how = 'left', on = 'fipscode')

# Hierarchical clustering? 
'''
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
plt.figure(figsize=(20, 7))  
plt.title("Dendrogram")  
shc.dendrogram(shc.linkage(data_imputed, method='ward'))
# shc.dendrogram(shc.linkage(data_complete, method='ward'))
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  
cluster.fit_predict(data_imputed)
# cluster.fit_predict(data_complete)
'''

# Look for interesting groupings (clusters by state?)
(pd.merge(ids, data_imputed[['fipscode', 'cluster_label']], how = 'left', on = 'fipscode')
 .groupby(['state', 'cluster_label']).count())

# Re-run with global mean? 
for i in data_prelim.loc[:, 'v001_rawvalue':'v156_rawvalue'].columns:
    data_prelim[i] = (data_prelim[i] - np.mean(data_prelim[i])) / np.std(data_prelim[i])
data_prelim = data_prelim.fillna(0).iloc[:, np.r_[2, 6:42]]
lower_limit = 1
upper_limit = 11
sse_rerun = {}
for k in range(lower_limit, upper_limit):
    k_means = KMeans(n_clusters = k, init = 'k-means++', n_init = 10, max_iter = 300, random_state = 123)
    k_means.fit(data_prelim)
    sse_rerun[k] = k_means.inertia_
# Set up multiple plots
figure, (plot1, plot2, plot3) = plt.subplots(1, 3, sharex = 'all', sharey = 'all')
# Plot 1
plot1.plot(range(lower_limit, upper_limit), sse_imputed.values())
plot1.set_xlabel('Clusters')
plot1.set_ylabel('Sum of Squared Distances')
plot1.set_xticks(range(lower_limit, upper_limit, 1))
plot1.set_title('Imputation with 0 (Average)')
# Plot 2
plot2.plot(range(lower_limit, upper_limit), sse_complete.values())
plot2.set_xlabel('Clusters')
plot2.set_xticks(range(lower_limit, upper_limit, 1))
plot2.set_title('Complete Cases, No Outliers')
# Plot 3
plot3.plot(range(lower_limit, upper_limit), sse_rerun.values())
plot3.set_xlabel('Clusters')
plot3.set_xticks(range(lower_limit, upper_limit, 1))
plot3.set_title('Rerun without Local/Regional Imputation')
plt.show()
# Display which counties are grouped together
k_means = KMeans(n_clusters = 4, init = 'k-means++', n_init = 10, max_iter = 300, random_state = 123)
k_means.fit(data_prelim)
data_prelim['cluster_label'] = k_means.labels_

(pd.merge(ids, data_prelim[['fipscode', 'cluster_label']], how = 'left', on = 'fipscode')
 .groupby(['state', 'cluster_label']).count())

(pd.merge(ids, data_imputed[['fipscode', 'cluster_label']], how = 'left', on = 'fipscode')
 .groupby(['state', 'cluster_label']).count())

# How many unique states per cluster?
(pd.merge(data_imputed, ids, how = 'left', on = 'fipscode').
 groupby('cluster_label')['state'].nunique())

# Did distance vary significantly by cluster? 
for i in range(0, 4):
    data_imputed['cluster_distance_space' + str(i)] = k_means.fit_transform(data_imputed)[0:, i]
data_imputed['min_distance'] = k_means.fit_transform(data_imputed).min(axis = 1)
data_imputed['max_distance'] = k_means.fit_transform(data_imputed).max(axis = 1)


data.fipscode[data.min_distdance == data_imputed.groupby('cluster_label')['min_distance'].min()]

# Which counties were closest together? 
'''
pd.merge(ids, data_imputed[['fipscode', 'cluster_label']], how = 'left', on = 'fipscode')



data_imputed.groupby('cluster_label')['fipscode', 'cluster_distance_space0'].agg(min)

pd.merge(data_imputed.groupby('cluster_label')['cluster_distance_space0'].agg(max),
         how = 'left', on = 'county')
# Which counties were in the same cluster but were furthest apart? 
                                 
# Do we need training and test sets
'''