import pandas as pd
import numpy as np
import matplotlib as mpl 
import scipy
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2

nb_sims = 10**6 # sample size
df = 5          # degrees of freedom (for student and chi-square distributions)
dist_name = 'normal' # student, normal, exponential, uniform, chi-square
dist_type = 'simulated RV' # real, custom

# TODO: add gamma distribution
if dist_name == 'normal':
    x = np.random.standard_normal(nb_sims)
    x_description = dist_type + ': ' + dist_name
elif dist_name == 'exponential':
    x = np.random.standard_exponential(nb_sims)
    x_description = dist_type + ': ' + dist_name
elif dist_name == 'uniform':
    x = np.random.uniform(0,1,nb_sims)
    x_description = dist_type + ': ' + dist_name
elif dist_name == 'student':
    x = np.random.standard_t(df=df,size=nb_sims)
    x_description = dist_type + ': ' + dist_name + ' | df = ' + str(df)
elif dist_name == 'chi-square':
    x = np.random.chisquare(df=df,size=nb_sims)
    x_description = dist_type + ': ' + dist_name + ' | df = ' + str(df)

'''
Goal: create a Jarque-Bera normality test
'''

x_mean = np.mean(x)
x_std = np.std(x)
x_skew = skew(x)
x_excess_kurtosis = kurtosis(x) # excess kurtosis
x_jb_stat = nb_sims/6*(x_skew**2 + 1/4*x_excess_kurtosis**2)
x_p_value = 1 - chi2.cdf(x_jb_stat, df=2) 
x_is_normal = (x_p_value > 0.05) # equivalently jb < 6

print(x_description)
print('mean is ' + str(x_mean))
print('standard deviation is ' + str(x_std))
print('skewness is ' + str(x_skew))
print('excess kurtosis is ' + str(x_excess_kurtosis))
print('JB statistic is ' + str(x_jb_stat))
print('p-value is ' + str(x_p_value))
print('is normal ' + str(x_is_normal))

# plot histogram
plt.figure()
plt.hist(x,bins=100) # bins defines the intervals of the histogram on the x axis
plt.title(x_description)
plt.show()
