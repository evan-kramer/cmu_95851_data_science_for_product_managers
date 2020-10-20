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

# Load data
dds9_clean = pd.read_csv('dds9_clean.csv')
dds10_clean = pd.read_csv('dds10_clean.csv')
dds11_clean = pd.read_csv('dds11_clean.csv')
dds_long = pd.concat([dds9_clean.assign(year = 2009), 
                      dds10_clean.assign(year = 2010),
                      dds11_clean.assign(year = 2011)])

# Convert categorical to numeric
dds11_clean = dds11_clean.astype('category')

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

# Check for outliers - not relevant for predictors of interest
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
                plt.xticks(np.arange(2009, 2012))
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
plt.xticks(np.arange(2009, 2012))
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
plt.xticks(np.arange(2009, 2012))
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
plt.xticks(np.arange(2009, 2012))
plt.yticks(np.arange(0, 101, 10))
plt.legend()
plt.savefig('Visualizations/cord_cutting.png', dpi = 900)

# Perceptions of ads
temp1 = {}
for v in dds11_clean.columns[dds11_clean.columns.str.contains('_ads')]:
    p = dds11_clean[v].value_counts().sort_index() / dds11_clean[v].count()
    print(v, ':', p[p.index.str.contains('Agree')].sum())
    temp1[v] = p[p.index.str.contains('Agree')].sum() * 100
temp1 = pd.DataFrame(temp1, index = [0]).transpose().rename(columns = {0: 'pct_agree'})
fig, ax = plt.subplots()
ax.barh(temp1.index, temp1.pct_agree, align = 'center')
ax.set_xticks(range(0, 101, 10))
ax.set_yticks(temp1.index)
ax.set_yticklabels(['Would view ads to pay less for video streaming',
                    'Would give personal info for targeted ads',
                    'Would pay to avoid ads while watching movies',
                    'Would pay to avoid ads while watching TV',
                    'Would pay to avoid ads while streaming music',
                    'Would pay to avoid ads while playing games',
                    'Would pay to avoid ads while watching sports',
                    'Would pay to avoid ads while reading news'])
ax.set_title('')
plt.savefig('Visualizations/avoid_ads.png', dpi = 900)

# Unbalanced classes
for v in ['age_cat', 'gender', 'region', 'employment_status', 'race_ethnicity',
          'income_cat', 'plan_to_purchase_smartwatch', 'plan_to_purchase_fitness_band']:
    fig, ax = plt.subplots()
    ax.barh((dds11_clean[v].value_counts() / dds11_clean[v].count()).index,
            dds11_clean[v].value_counts() / dds11_clean[v].count() * 100)
    ax.set_xticks(range(0, 101, 10))
    ax.set_title(v.replace('_cat', '')
                 .replace('_', ' ')
                 .replace('ce Ethn', 'ce/Ethn')
                 .title())
    ax.set_xlabel('Percentage')
    plt.savefig('Visualizations/' + v + '.png', dpi = 900)

# 
'''
category_names = ['Strongly disagree', 'Disagree',
                  'Neither agree nor disagree', 'Agree', 'Strongly agree']
results = {
    'Question 1': [10, 15, 17, 32, 26],
    'Question 2': [26, 22, 29, 10, 13],
    'Question 3': [35, 37, 7, 2, 19],
    'Question 4': [32, 11, 9, 15, 33],
    'Question 5': [21, 29, 5, 5, 40],
    'Question 6': [8, 19, 5, 30, 38]
}


def survey(results, category_names):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn')(
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5,
                label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            ax.text(x, y, str(int(c)), ha='center', va='center',
                    color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    return fig, ax


survey(results, category_names)
plt.show()
'''