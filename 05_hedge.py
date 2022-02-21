import pandas as pd
import numpy as np
import matplotlib as mpl 
import scipy
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2, linregress

# Import our own files and reload
import file_classes
importlib.reload(file_classes)
import file_functions
importlib.reload(file_functions)

# Input parameters
benchmark = '^STOXX50E'
security = 'BBVA.MC' # or ric, Reuters Instruent Code (ticker-like code used by Refinitiv to identify financial instruments and indices)
hedge_rics = ['SAN.MC', 'REP.MC']
delta_portfolio = 10 # mln USD

# Compute betas
capm = file_classes.capm_manager(benchmark, security)
capm.load_timeseries()
capm.compute()
beta_portfolio = capm.beta
beta_portfolio_usd = beta_portfolio * delta_portfolio # mln USD



