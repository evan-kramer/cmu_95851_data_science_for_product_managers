# -*- coding: utf-8 -*-
'''
Evan Kramer
evankram
10/23/2020
'''
# Import modules
import pandas as pd
import numpy as np
import re

# Load data
dds9 = pd.read_excel('Deloitte_Digital_Democracy_data/DDS9_Data_Extract_with_labels.xlsx')
dds10 = pd.read_excel('Deloitte_Digital_Democracy_data/DDS10_Data_Extract_with_labels.xlsx')
dds11 = pd.read_excel('Deloitte_Digital_Democracy_data/DDS11_Data_Extract_with_labels.xlsx')

# Clean up names
names9 = []
for i in dds9.columns:
    try:
        names9.append(re.search('- (.*?)$', i).group(1))
    except:
        names9.append(i)
names10 = []
for i in dds10.columns:
    try:
        names10.append(re.search('- (.*?)$', i).group(1))
    except:
        names10.append(i)
names11 = []
for i in dds11.columns:        
    try:
        names11.append(re.search('- (.*?)$', i).group(1))        
    except:
        names11.append(i)
        
# Compare fields and then join/combine
pd.concat([
    # DDS9
    (pd.DataFrame([dds9.columns])
     .transpose()
     .rename(columns = {0:'varname'})
     .assign(file = 'DDS9', 
             varname_clean = names9)),
    # DDS10
    (pd.DataFrame([dds10.columns])
     .transpose()
     .rename(columns = {0:'varname'})
     .assign(file = 'DDS10', 
             varname_clean = names10)),
    # DDS11
    (pd.DataFrame([dds11.columns])
     .transpose()
     .rename(columns = {0:'varname'})
     .assign(file = 'DDS11', 
             varname_clean = names11)),
    ]).to_excel('varnames.xlsx', index = False)

names_clean = pd.read_excel('varnames_clean.xlsx')
names_clean9 = pd.merge(pd.DataFrame(names9),
                        names_clean.varname_short[names_clean.varname_clean9.notnull()].reset_index().drop(columns = 'index'),
                        how = 'left', left_index = True, right_index = True)
names_clean10 = pd.merge(pd.DataFrame(names10),
                         names_clean.varname_short[names_clean.varname_clean10.notnull()].reset_index().drop(columns = 'index'),
                         how = 'left', left_index = True, right_index = True)
names_clean11 = pd.merge(pd.DataFrame(names11),
                         names_clean.varname_short[names_clean.varname_clean11.notnull()].reset_index().drop(columns = 'index'),
                         how = 'left', left_index = True, right_index = True)

# Output clean files
(dds9.rename(columns = {dds9.columns[i]:names_clean9.varname_short[i] for i in range(len(names9))})
 .to_csv('dds9_clean.csv', index = False))
(dds10.rename(columns = {dds10.columns[i]:names_clean10.varname_short[i] for i in range(len(names10))})
 .to_csv('dds10_clean.csv', index = False))
(dds11.rename(columns = {dds11.columns[i]:names_clean11.varname_short[i] for i in range(len(names11))})
 .to_csv('dds11_clean.csv', index = False))

# For DDS9, make small/large smartphone/tablet and laptop/tablet hybrid questions all one
dds9_clean = pd.read_csv('dds9_clean.csv')
(dds9_clean
 .assign(
     own_laptop = np.where((dds9_clean.own_laptop == 'Yes') | (dds9_clean.own_laptop_tablet_hybrid == 'Yes'),
                           'Yes', 'No'),
     plan_to_purchase_laptop = np.where((dds9_clean.plan_to_purchase_laptop == 'Yes') | (dds9_clean.plan_to_purchase_laptop_tablet_hybrid == 'Yes'),
                                         'Yes', 'No'),
     value_rank_laptop = np.where(dds9_clean.value_rank_laptop > dds9_clean.value_rank_laptop_tablet_hybrid, 
                                  dds9_clean.value_rank_laptop, dds9_clean.value_rank_laptop_tablet_hybrid),
     own_tablet = np.where((dds9_clean.own_tablet == 'Yes') | (dds9_clean.own_tablet_small == 'Yes'),
                           'Yes', 'No'),
     plan_to_purchase_tablet = np.where((dds9_clean.plan_to_purchase_tablet == 'Yes') | (dds9_clean.plan_to_purchase_tablet_small == 'Yes'),
                                        'Yes', 'No'),
     value_rank_tablet = np.where(dds9_clean.value_rank_tablet > dds9_clean.value_rank_tablet_small, 
                                  dds9_clean.value_rank_tablet, dds9_clean.value_rank_tablet_small),
     own_smartphone = np.where((dds9_clean.own_smartphone == 'Yes') | (dds9_clean.own_smartphone_large == 'Yes'), 
                                     'Yes', 'No'),
     plan_to_purchase_smartphone = np.where((dds9_clean.plan_to_purchase_smartphone == 'Yes') | (dds9_clean.plan_to_purchase_smartphone_large == 'Yes'),
                                            'Yes', 'No'),
     value_rank_smartphone = np.where(dds9_clean.value_rank_smartphone > dds9_clean.value_rank_smartphone_large,
                                      dds9_clean.value_rank_smartphone, dds9_clean.value_rank_smartphone_large)
 )
 .drop(columns = dds9_clean.columns[dds9_clean.columns.str.contains('hybrid|large|small')])
).to_csv('dds9_clean.csv', index = False)

# For DDS11, convert str to categorical variables for random forest classifier
dds11_clean = pd.read_csv('dds11_clean.csv')

# Convert categorical to numeric
dds11_clean = dds11_clean.astype('category')
for v in dds11_clean.columns: 
    if dds11_clean[v].value_counts().sort_index() == 
dds11_clean.to_csv('dds11_clean.csv', index = False)