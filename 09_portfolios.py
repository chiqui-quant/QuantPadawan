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

notional = 10 # mln USD
rics = ['BBVA.MC', 'AAL.L', 'ANTO.L', 'EONGn.DE', 'RIO.L']

port_mgr = file_classes.portfolio_manager(rics, notional)
port_mgr.compute_covariance_matrix(bool_print=False)

port_min_var = port_mgr.compute_portfolio('min-variance')
x_min_var = port_min_var.weights
port_min_var.summary()

port_pca = port_mgr.compute_portfolio('pca')
x_pca = port_pca.weights
port_pca.summary()

port_equi = port_mgr.compute_portfolio('equi-weight')
x_equi = port_equi.weights
port_equi.summary()