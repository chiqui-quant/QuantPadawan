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
importlib.reload(file_functions)


# Inputs
benchmark = '^STOXX50E' # variable x
security = 'BBVA.MC' # variable y

# Load 2 timeseries
table_x = file_functions.load_timeseries(benchmark)
table_y = file_functions.load_timeseries(security)
# Note: we might have a different number of observations, we need to synchronize the timeseries

# Synchronize timeseries
timestamps_x = list(table_x['date'].values)
timestamps_y = list(table_y['date'].values)
timestamps = list(set(timestamps_x) & set(timestamps_y))

# Sinchronized timeseries for benchmark
table_x_sync = table_x[table_x['date'].isin(timestamps)]
table_x_sync.sort_values(by='date', ascending=True)
table_x_sync = table_x_sync.reset_index(drop=True)

# Sinchronized timeseries for security
table_y_sync = table_y[table_y['date'].isin(timestamps)]
table_y_sync.sort_values(by='date', ascending=True)
table_y_sync = table_y_sync.reset_index(drop=True)

# Table of returns for benchmark and security
t = pd.DataFrame()
t['date'] = table_x_sync['date']
t['price_x'] = table_x_sync['close']
t['return_x'] = table_x_sync['return_close']
t['price_y'] = table_y_sync['close']
t['return_y'] = table_y_sync['return_close']
