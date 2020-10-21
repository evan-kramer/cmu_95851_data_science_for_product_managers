# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 20:18:04 2020

@author: evan.kramer
"""

'''
for v in dds11_clean.columns:
    try:
        float(dds11_clean[v])
    except: 
        # r = pd.DataFrame(dds11_clean[v].value_counts().sort_values()).reset_index()
        # r2 = {r['index'][i]: i for i in range(len(r))}
        # dds11_clean[v] = dds11_clean[v].map(r2)
        dds11_clean[v] = dds11_clean[v].astype('category')
        # print(dds11_clean[v].value_counts())
'''