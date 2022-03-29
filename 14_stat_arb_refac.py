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
backtest = file_classes.backtest()
backtest.ric_long = 'SAN.MC'
backtest.ric_short = 'BBVA.MC'
backtest.rolling_days = 20
backtest.level_1 = 1.
backtest.level_2 = 2.
backtest.data_cut = 0.7
backtest.data_type = 'in-sample' # in-sample out-of-sample

# Load data
backtest.load_data()

# Compute indicator
backtest.compute_indicator(bool_plot=True)

# Run backtest of trading strategy
backtest.run_strategy(bool_plot=True)

# Get results in a dataframe format
df_strategy = backtest.dataframe_strategy