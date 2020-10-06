# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 18:15:35 2020

@author: evan.kramer
"""
import os
import numpy as np
import pandas as pd
os.chdir("C:/Users/evan.kramer/Documents/CMU/Courses/2020-01/95851 - Making Products Count, Data Science for Product Managers/Assignments/Python Proficiency Exercise")

# The data set for this exercise is found in the file Baseball Data. 
# It shows the outcome of the 2016 Fantasy Baseball League. 
# Your tasks are as follows:
# 1.Import the database as a Pandas dataframe.
baseball = pd.read_excel("Baseball Data.xlsx")
print(baseball.columns)
# a.Which player had the highest batting average? The lowest? Make sure to print out your results.
print(baseball.PLAYER[baseball['Batting Average'] == max(baseball['Batting Average'])])
print(baseball.PLAYER[baseball['Batting Average'] == min(baseball['Batting Average'])])
# b.Which player had the highest # of home-runs per game?
baseball['hr_per_g'] = baseball['Home Runs'] / baseball['Games Played']
print(baseball.PLAYER[baseball['hr_per_g'] == max(baseball['hr_per_g'])])
# c.Do players with higher batting averages tend to score more home runs per game? Create a scatter plot and determine if a relationship exists (make sure to include labels!).
import matplotlib.pyplot as plt
plot = plot.figure() 
plt.scatter(
    x = baseball['Batting Average'], 
    y = baseball['Home Runs'], 
    c = "gray", 
    alpha = 0.3
    )

# 2.Create a numpy array of Games Played.

# a.What is the mean number of games played? The median? 
# b.Plot a histogram of this data, choosing an appropriate bin size, and observe the distribution.

# 3.Isolate players who play on 1st Base, and those who play outfield.
# a.To which position does the highest home-run scoring player belong to?
# b.Compare the means and medians of batting averages. Can you conclude that one group hits more successfully than the other?

# 4.Submit your work as a Jupyter notebook (.ipynb file)