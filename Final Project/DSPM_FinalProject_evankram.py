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
(dds9.rename(columns = {dds9.columns[i]:names_clean9.varname_short[i] for i in range(len(names9))})
 .to_csv('dds9_clean.csv', index = False))
(dds10.rename(columns = {dds10.columns[i]:names_clean10.varname_short[i] for i in range(len(names10))})
 .to_csv('dds10_clean.csv'))
(dds11.rename(columns = {dds11.columns[i]:names_clean11.varname_short[i] for i in range(len(names11))})
 .to_csv('dds11_clean.csv'))




# Concatenate data the data are cross-sectional (2009, 2010, 2011)
# Crosswalk names
# Design question
# Exploratory data analysis
# How have ownership/plan to purchase rates changed over time?


# Plots



# True for own, plan to purchase, and value_rank
# Will need to combine tablet small and large (2009 has them separate, 2010 combined)
# Tablet is one category instead of three in 2010 (unlike 2009)
# Same thing with smartphone; large and small is one in 2010, but not in 2009
# Digital antenna is a new one for 2010 as well
# Smart glasses in 2009 (not in 2010), VR headset in 2010 (not in 2009)
# 3d printer in 2009 (not in 2010), drone in 2010 (not in 2009)
# Digital TV antenna in 2010 but not in 2009
# Add app use dating, messaging, mobile payment, education, tickets, reservations, hobbies in 2010
# Add tv_on_demand in place of tv_rent_download
# Transform rank values to 1-3? 

# Questions vary by year; combine, drop, and standardize

