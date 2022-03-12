import pandas as pd
import numpy as np
import matplotlib as mpl 
import scipy
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2

# Import our own files and reload
import file_classes
importlib.reload(file_classes)
import file_functions

# Inputs
backtest = file_classes.backtest()
backtest.ric_long = 'SAN.MC'
backtest.ric_short = 'BBVA.MC'
backtest.rolling_days = 20
backtest.level_1 = 1.
backtest.level_2 = 2.
backtest.data_cut = 0.7
backtest.data_type = 'in-sample' # in-sample out-of-sample

# Load data
t = file_functions.load_synchronized_timeseries(backtest.ric_long, backtest.ric_short)
cut = int(backtest.data_cut*t.shape[0])
if backtest.data_type == 'in-sample':
    df1 = t.head(cut)
elif backtest.data_type == 'out-of-sample':
    df1 = t.tail(t.shape[0]-cut)
df1 = df1.reset_index(drop=True)

# Spread at current close
df1['spread'] = df1['price_x']/df1['price_y'] # Note: the stocks should be quoted in the same currency
base = df1['spread'][0] # normalize spread so we can compare different pairs of stocks
df1['spread'] = df1['spread'] / base

# Spread at previous close
df1['spread_previous'] = df1['price_x_previous']/df1['price_y_previous'] 
df1['spread_previous'] = df1['spread_previous'] / base

# Compute bollinger bands
size = df1.shape[0]
columns = ['lower_2','lower_1', 'mean', 'upper_1', 'upper_2']
mtx_bollinger = np.empty((size, len(columns)))
mtx_bollinger[:] = np.nan
for n in range(backtest.rolling_days-1,size):
    vec_price = df1['spread'].values
    vec_price = vec_price[n-backtest.rolling_days+1:n+1] # to check
    mu = np.mean(vec_price)
    sigma = np.std(vec_price)
    m = 0
    mtx_bollinger[n][m] = mu - backtest.level_2*sigma
    m = m+1
    mtx_bollinger[n][m] = mu - backtest.level_1*sigma
    m = m+1
    mtx_bollinger[n][m] = mu 
    m = m+1
    mtx_bollinger[n][m] = mu + backtest.level_1*sigma
    m = m+1
    mtx_bollinger[n][m] = mu + backtest.level_2*sigma
    m = m+1
df2 = pd.DataFrame(data=mtx_bollinger, columns=columns)
timeseries = pd.concat([df1,df2], axis=1) # axis=0 for rows, axis=1 for columns
timeseries = timeseries.dropna()
timeseries = timeseries.reset_index(drop=True)

# Plot
timestamps = timeseries['date']
spread = timeseries['spread']
mu = timeseries['mean']
u1 = timeseries['upper_1']
u2 = timeseries['upper_2']
l1 = timeseries['lower_1']
l2 = timeseries['lower_2']
plt.figure()
plt.title('Spread ' + backtest.ric_long + ' / ' + backtest.ric_short)
plt.xlabel('Time')
plt.ylabel('Price')
plt.plot(timestamps, mu, color='blue', label='mean')
plt.plot(timestamps, l1, color='green', label='lower_1')
plt.plot(timestamps, u1, color='green', label='upper_1')
plt.plot(timestamps, l2, color='red', label='lower_2')
plt.plot(timestamps, u2, color='red', label='upper_2')
plt.plot(timestamps, spread, color='black', label='spread')
plt.legend(loc=0)
plt.grid()
plt.show()