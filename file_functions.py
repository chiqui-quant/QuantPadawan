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
