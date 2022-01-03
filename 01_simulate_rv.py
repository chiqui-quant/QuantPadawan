# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 21:12:50 2021

@author: Chiqui
"""

import pandas as pd
import numpy as np
import matplotlib as mpl 
import scipy
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2

nb_sims = 10**6
df = 2
dist_name = 'chi-square' # student normal exponential uniform chi-square
dist_type = 'simulated RV' # real custom

if dist_name == 'normal':
    x = np.random.standard_normal(nb_sims)
    x_description = dist_type + ' ' + dist_name
elif dist_name == 'exponential':
    x = np.random.standard_exponential(nb_sims)
    x_description = dist_type + ' ' + dist_name
elif dist_name == 'uniform':
    x = np.random.uniform(0,1,nb_sims)
    x_description = dist_type + ' ' + dist_name
elif dist_name == 'student':
    x = np.random.standard_t(df=df,size=nb_sims)
    x_description = dist_type + ' ' + dist_name + ' | df = ' + str(df)
elif dist_name == 'chi-square':
    x = np.random.chisquare(df=df,size=nb_sims)
    x_description = dist_type + ' ' + dist_name + ' | df = ' + str(df)

'''
Goal: create a Jarque-Bera normality test
'''

x_skew = skew(x)
x_kurtosis = kurtosis(x)


# plot histogram
plt.figure()
plt.hist(x,bins=100) # bins is the number of segments
plt.title(x_description)
plt.show()