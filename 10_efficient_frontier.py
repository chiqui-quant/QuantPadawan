import pandas as pd
import numpy as np
import matplotlib as mpl 
import scipy
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2, linregress
from scipy.optimize import minimize

# Import our own files and reload
import file_classes
importlib.reload(file_classes)
import file_functions
importlib.reload(file_functions)

# Universe
rics = ['BBVA.MC', 'AAL.L', 'ANTO.L', 'EONGn.DE', 'RIO.L']

# Input parameters
notional = 300 # mlnUSD
target_return = 0.0 
include_min_var = True

# Efficient frontier
dict_portfolios = file_functions.compute_efficient_frontier(rics, notional, target_return, include_min_var)