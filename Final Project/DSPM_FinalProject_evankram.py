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
dds9 = pd.read_excel('Deloitte_Digital_Democracy_data/DDS9_Data_Extract_with_labels.xlsx')

# Concatenate data the data are cross-sectional (2009, 2010, 2011)
# Crosswalk names
# Design question
# Exploratory data analysis
# Plots
for c in dds9.columns:
    print(dds9[c].value_counts())
    plt.hist(dds9[c])
    plt.title(c)
    plt.show()