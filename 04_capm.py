import pandas as pd
import numpy as np
import matplotlib as mpl 
import scipy
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2, linregress

# Import our own files and reload
import file_classes
importlib.reload(file_classes)
import file_functions
importlib.reload(file_functions)


# Inputs
benchmark = '^STOXX50E' # variable x
security = 'BBVA.MC' # variable y
nb_decimals = 4

# Load synchronized timeseries
t = file_functions.load_synchronized_timeseries(ric_x=benchmark, ric_y=security)

# Linear regression
x = t['return_x'].values
y = t['return_y'].values
slope, intercept, r_value, p_value, std_err = linregress(x,y)
slope = np.round(slope, nb_decimals)
intercept = np.round(intercept, nb_decimals)
p_value = np.round(p_value, nb_decimals)
null_hypothesis = p_value > 0.05 # if p_value < 0.05 reject the null hypothesis
r_value = np.round(r_value, nb_decimals) # correlation coefficient
r_squared = np.round(r_value**2, nb_decimals)
predictor_linreg = intercept + slope*x

# Scatterplot of returns
str_title = 'Scatterplot of returns' + '\n'\
    + 'Linear regression | security ' + security\
    + ' | benchmark ' + benchmark + '\n'\
    + 'alpha (intercept) ' + str(intercept)\
    + ' | beta (slope) ' + str(slope) + '\n'\
    + 'p_value ' + str(p_value)\
    + ' | null hypothesis ' + str(null_hypothesis) + '\n'\
    + 'r-value (corr) ' + str(r_value)\
    + ' | r-squared ' + str(r_squared)
plt.figure() 
plt.title(str_title)
plt.scatter(x,y)
plt.plot(x, predictor_linreg, color='green')
plt.xlabel(benchmark)
plt.ylabel(security)
plt.grid()
plt.show()   


