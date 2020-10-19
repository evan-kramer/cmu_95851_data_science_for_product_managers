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
dds9_clean = pd.read_csv('dds9_clean.csv')
dds10_clean = pd.read_csv('dds10_clean.csv')
dds11_clean = pd.read_csv('dds11_clean.csv')
dds_long = pd.concat([dds9_clean.assign(year = 9), 
                      dds10_clean.assign(year = 10),
                      dds11_clean.assign(year = 11)])

# Compute summary statistics
dds11_clean.describe()

# Check for and deal with missing values 
# No missing values
for v in dds11_clean.columns: 
    try:
        print(v, ': ', dds11_clean[v][np.isnan(dds11_clean[v])].count(), 
              ' missing values', sep = '')
    except:
        print(v, ': ', dds11_clean[v][dds11_clean[v].isnull()].count(), 
              ' missing values', sep = '') 

# Check for outliers
for v in dds11_clean.columns:
    try:
        plt.hist(dds11_clean[v])
        plt.title(v.replace('_', ' ').title())
        plt.xticks(rotation='vertical')
        plt.show()
    except:
        pass

# How have ownership rates, plan to purchase, etc. changed over time? 
# Look at rates over time for yes/no responses
pct_changes = {}
abs_changes = {}
for v in dds_long.columns:
    try:
        if dds_long[v].value_counts().index.sort_values().all(True) == 'Yes':
            temp = dds_long.groupby('year')[v].value_counts().unstack()
            if len(temp) == 1:
                pass
            else:
                # Create total
                temp['Total'] = temp.Yes + temp.No
                for v2 in ['Yes', 'No']:
                    temp[v2] = 100 * temp[v2] / (temp.Total)                    
                
                # Largest swings
                pct_changes[v] = (100 * (temp.Yes[max(temp.index)] - temp.Yes[min(temp.index)]) / 
                              temp.Yes[min(temp.index)])
                abs_changes[v] = (temp.Yes[max(temp.index)] - temp.Yes[min(temp.index)])
                
                # Line plot
                plt.plot(temp.index, temp.Yes, marker = 'o')
                plt.xticks(np.arange(9, 12))
                plt.yticks(np.arange(0, 101, 10))
                plt.title('Percent Responding "Yes" to \n ' + v.replace('_', ' ').title())
                plt.show()
        else:
            pass
    except:
        print("There was a problem with", v)

# Biggest year-over-year changes
absolute_changes = (pd.DataFrame([abs_changes.keys(), abs_changes.values()])
 .transpose()
 .rename(columns = {0: 'varname', 1: 'absolute_change'})
 .sort_values('absolute_change', ascending = False))
absolute_changes[absolute_changes.varname.str.contains('plan_to_purchase')]

relative_changes = (pd.DataFrame([pct_changes.keys(), pct_changes.values()])
 .transpose()
 .rename(columns = {0: 'varname', 1: 'percent_change'})
 .sort_values('percent_change', ascending = False))
relative_changes[relative_changes.varname.str.contains('plan_to_purchase')]

# So we'll focus on wearables, given that there's the greatest increase in interest
# Descriptives/distribution by age, region, income, race/ethnicity
for v in ['age_cat', 'gender', 'region', 'employment_status', 'race_ethnicity']:
    fig, ax = plt.subplots()
    ax.barh(dds11_clean[v].value_counts().index,
            dds11_clean[v].value_counts().values,
            align = 'center')
    ax.set_yticks(dds11_clean[v].value_counts().index)
    ax.set_title(v.replace('_', ' ').title())
    plt.show()

# Check for balanced classes

# Visualizations
# Rise of wearables
temp1 = dds_long.groupby('year')['plan_to_purchase_smartwatch'].value_counts().unstack()
temp1['Total'] = temp1.Yes + temp1.No
temp1['Yes'] = temp1.Yes / temp1.Total * 100
temp1['No'] = temp1.No / temp1.Total * 100
temp2 = dds_long.groupby('year')['own_smartwatch'].value_counts().unstack()
temp2['Total'] = temp2.Yes + temp2.No
temp2['Yes'] = temp2.Yes / temp2.Total * 100
temp2['No'] = temp2.No / temp2.Total * 100

plt.plot(temp1.index, temp1.Yes, marker = 'o', label = 'Plan to purchase smartwatch')
plt.plot(temp2.index, temp2.Yes, marker = 'o', label = 'Own smartwatch')
plt.xticks(np.arange(9, 12))
plt.yticks(np.arange(0, 101, 10))
plt.legend()
plt.savefig('Visualizations/smartwatch.png', dpi = 900)

temp1 = dds_long.groupby('year')['plan_to_purchase_fitness_band'].value_counts().unstack()
temp1['Total'] = temp1.Yes + temp1.No
temp1['Yes'] = temp1.Yes / temp1.Total * 100
temp1['No'] = temp1.No / temp1.Total * 100
temp2 = dds_long.groupby('year')['own_fitness_band'].value_counts().unstack()
temp2['Total'] = temp2.Yes + temp2.No
temp2['Yes'] = temp2.Yes / temp2.Total * 100
temp2['No'] = temp2.No / temp2.Total * 100

plt.plot(temp1.index, temp1.Yes, marker = 'o', label = 'Plan to purchase fitness band')
plt.plot(temp2.index, temp2.Yes, marker = 'o', label = 'Own fitness band')
plt.xticks(np.arange(9, 12))
plt.yticks(np.arange(0, 101, 10))
plt.legend()
plt.savefig('Visualizations/fitness_band.png', dpi = 900)

# Cord cutting
temp1 = dds_long.groupby('year')['own_tv'].value_counts().unstack()
temp1['Total'] = temp1.Yes + temp1.No
temp1['Yes'] = temp1.Yes / temp1.Total * 100
temp1['No'] = temp1.No / temp1.Total * 100
temp2 = dds_long.groupby('year')['subscription_cable'].value_counts().unstack()
temp2['Total'] = temp2.Yes + temp2.No
temp2['Yes'] = temp2.Yes / temp2.Total * 100
temp2['No'] = temp2.No / temp2.Total * 100
temp3 = dds_long.groupby('year')['subscription_internet'].value_counts().unstack()
temp3['Total'] = temp3.Yes + temp3.No
temp3['Yes'] = temp3.Yes / temp3.Total * 100
temp3['No'] = temp3.No / temp3.Total * 100
temp4 = dds_long.groupby('year')['plan_to_purchase_smartphone'].value_counts().unstack()
temp4['Total'] = temp4.Yes + temp4.No
temp4['Yes'] = temp4.Yes / temp4.Total * 100
temp4['No'] = temp4.No / temp4.Total * 100

plt.plot(temp1.index, temp1.Yes, marker = 'o', label = 'Own a TV')
plt.plot(temp2.index, temp2.Yes, marker = 'o', label = 'Have cable/satellite')
plt.plot(temp3.index, temp3.Yes, marker = 'o', label = 'Have internet subscription')
plt.plot(temp4.index, temp4.Yes, marker = 'o', label = 'Plan to purchase smartphone')
plt.xticks(np.arange(9, 12))
plt.yticks(np.arange(0, 101, 10))
plt.legend()
plt.savefig('Visualizations/cord_cutting.png', dpi = 900)


