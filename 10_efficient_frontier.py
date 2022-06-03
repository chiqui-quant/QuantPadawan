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
rics = ['A2A.MI', 'AMP.MI', 'ATL.MI', 'AZM.MI', 'IP.MI']

# Input parameters
notional = 10 # mln EUR
target_return = 0.16
include_min_var = False

# Efficient frontier
dict_portfolios = file_functions.compute_efficient_frontier(rics, notional, target_return, include_min_var)