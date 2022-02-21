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

def load_timeseries(ric):
    directory = 'C:\\Users\\Chiqui\\Desktop\\Python Projects\\QuantPadawan\\data\\'
    path = directory + ric + '.csv'
    raw_data = pd.read_csv(path)
    t = pd.DataFrame() # t (for table) with no elements
    t['date'] = pd.to_datetime(raw_data['Date'], dayfirst=True) # if dayfirst is omitted we have American dates
    t['close'] = raw_data['Close']
    t.sort_values(by='date',ascending=True)
    t['close_previous'] = t['close'].shift(1) # shift moves values below by the amount you indicate
    t['return_close'] = t['close']/t['close_previous'] - 1
    t = t.dropna()
    t = t.reset_index(drop=True)

    return t

def load_synchronized_timeseries(ric_x, ric_y):
    # Get timeseries of x and y
    table_x = file_functions.load_timeseries(ric_x)
    table_y = file_functions.load_timeseries(ric_y)
    # Note: we might have a different number of observations, we need to synchronize the timeseries
    # Synchronize timestamps
    timestamps_x = list(table_x['date'].values)
    timestamps_y = list(table_y['date'].values)
    timestamps = list(set(timestamps_x) & set(timestamps_y))
    # Sinchronized timeseries for x
    table_x_sync = table_x[table_x['date'].isin(timestamps)]
    table_x_sync.sort_values(by='date', ascending=True)
    table_x_sync = table_x_sync.reset_index(drop=True)
    # Sinchronized timeseries for y
    table_y_sync = table_y[table_y['date'].isin(timestamps)]
    table_y_sync.sort_values(by='date', ascending=True)
    table_y_sync = table_y_sync.reset_index(drop=True)
    # Table of returns for x and y
    t = pd.DataFrame()
    t['date'] = table_x_sync['date']
    t['price_x'] = table_x_sync['close']
    t['return_x'] = table_x_sync['return_close']
    t['price_y'] = table_y_sync['close']
    t['return_y'] = table_y_sync['return_close']

    return t

