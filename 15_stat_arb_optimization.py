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
backtest.data_cut = 0.1
backtest.data_type = 'in-sample' # in-sample out-of-sample

# First overview
list_rolling_days = [10,20,30,40,50,60]
list_level_1 = [0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0]

# We can play with these values and zoom in 
# list_rolling_days = [10,15,20,25,30,35]
# list_level_1 = [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4]

# Third zoom to find the optimal parameters
# list_rolling_days = [15,16,17,18,19,20,21,22,23,24,25]
# list_level_1 = [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4]

mtx_sharpe = np.zeros((len(list_rolling_days), len(list_level_1)))
mtx_nb_trades = np.zeros((len(list_rolling_days), len(list_level_1)))

rd = 0
for rolling_days in list_rolling_days:
    l1 = 0
    for level_1 in list_level_1:
        # Parameters to optimize
        backtest.rolling_days = rolling_days
        backtest.level_1 = level_1
        backtest.level_2 = 2*level_1 # symmetric take profit and stop loss 
        # Load data
        backtest.load_data()
        # Compute indicator
        backtest.compute_indicator()
        # Run backtest of trading strategy
        backtest.run_strategy()
        # Get results in a dataframe format
        df_strategy = backtest.dataframe_strategy
        # Save data in tables
        mtx_sharpe[rd][l1] = backtest.pnl_sharpe
        mtx_nb_trades[rd][l1] = backtest.nb_trades
        l1 += 1
    rd += 1

# Show different levels for the Sharpe matrix
df1 = pd.DataFrame()
df1['rolling_days'] = list_rolling_days
column_names = ['level_1_' + str(level_1) for level_1 in list_level_1] 
df2 = pd.DataFrame(data=mtx_sharpe, columns=column_names)
df_sharpe = pd.concat([df1,df2], axis=1) # axis=0 for rows, axis=1 for columns
df_sharpe = df_sharpe.reset_index(drop=True)

# Table of the number of trades
df1 = pd.DataFrame()
df1['rolling_days'] = list_rolling_days
column_names = ['level_1_' + str(level_1) for level_1 in list_level_1] 
df2 = pd.DataFrame(data=mtx_nb_trades, columns=column_names)
df_trades = pd.concat([df1,df2], axis=1) # axis=0 for rows, axis=1 for columns
df_trades = df_trades.reset_index(drop=True)
