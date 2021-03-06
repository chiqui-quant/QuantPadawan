import pandas as pd
import numpy as np
import matplotlib as mpl 
import scipy
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2, linregress
from scipy.optimize import minimize
from numpy import linalg as LA

# Import our own files and reload
import file_classes
importlib.reload(file_classes)
import file_functions
importlib.reload(file_functions)

notional = 10 # mln EUR
rics = ['A2A.MI', 'AMP.MI', 'ATL.MI', 'AZM.MI', 'BAMI.MI']

port_mgr = file_classes.portfolio_manager(rics, notional)
port_mgr.compute_covariance_matrix(bool_print=False)

port_min_var = port_mgr.compute_portfolio('min-variance') # Note: this is the analytical solution
port_min_var.summary()

port_min_var_l1 = port_mgr.compute_portfolio('min-variance-l1')
port_min_var_l1.summary()

port_min_var_l2 = port_mgr.compute_portfolio('min-variance-l2') # Note: this is a numerical solution
port_min_var_l2.summary()

port_long_only = port_mgr.compute_portfolio('long-only')
port_long_only.summary()

port_pca = port_mgr.compute_portfolio('pca')
port_pca.summary()

port_equi = port_mgr.compute_portfolio('equi-weight')
port_equi.summary()

port_volatility = port_mgr.compute_portfolio('volatility-weighted')
port_volatility.summary()

port_markowitz = port_mgr.compute_portfolio('markowitz', target_return=0.05)
port_markowitz.summary()