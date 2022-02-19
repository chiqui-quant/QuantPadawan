import pandas as pd
import numpy as np
import matplotlib as mpl 
import scipy
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2

# Get market data (remember to modify path to match your own directory)
directory = 'C:\\Users\\Chiqui\\Desktop\\Python Projects\\QuantPadawan\\data\\'

# Inputs
ric = 'BBVA.MC'
path = directory + ric + '.csv'
raw_data = pd.read_csv(path)

# Use prices
# x = raw_data['Close'].values
# x_description = 'market data ' + ric
# nb_rows = len(x) # alternatively nb_rows = raw_data.shape[0]

# Note: but we want to use returns for the normality test
# Create table of returns
t = pd.DataFrame() # t (for table) with no elements
t['date'] = pd.to_datetime(raw_data['Date'], dayfirst=True) # if dayfirst is omitted we have American dates
t['close'] = raw_data['Close']
t.sort_values(by='date',ascending=True)
t['close_previous'] = t['close'].shift(1) # shift moves values below by the amount you indicate
t['return_close'] = t['close']/t['close_previous'] - 1
t = t.dropna()
t = t.reset_index(drop=True)

x = t['return_close'].values
x_description = 'market data ' + ric
nb_rows = len(x)


'''
Goal: create a Jarque-Bera normality test
'''

x_mean = np.mean(x)
x_std = np.std(x)
x_skew = skew(x)
x_kurtosis = kurtosis(x) # excess kurtosis
x_jb_stat = nb_rows/6*(x_skew**2 + 1/4*x_kurtosis**2)
x_p_value = 1 - chi2.cdf(x_jb_stat, df=2) 
x_is_normal = (x_p_value > 0.05) # equivalently jb < 6


print('skewness is ' + str(x_skew))
print('kurtosis is ' + str(x_kurtosis))
print('JB statistic is ' + str(x_jb_stat))
print('p-value is ' + str(x_p_value))
print('is normal ' + str(x_is_normal))


# plot histogram
plt.figure()
plt.hist(x,bins=100) # bins is the number of segments
plt.title(x_description)
plt.show()