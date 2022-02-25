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

# Inputs
inputs = file_classes.hedge_input()
inputs.benchmark = '^STOXX50E'
inputs.security = 'BBVA.MC'
inputs.hedge_securities = ['^GDAXI', '^FCHI'] 
inputs.delta_portfolio = 10 # mln USD

# Computations
hedge = file_classes.hedge_manager(inputs)
hedge.load_betas()
# Exact solution
hedge.compute_exact()
optimal_hedge_exact = hedge.optimal_hedge
# Numerical solution
hedge.compute(regularization=0.0)
hedge_delta = hedge.hedge_delta
hedge_beta_usd = hedge.hedge_beta_usd

# # Numerical solution
# dimensions = len(inputs.hedge_securities)
# x = np.zeros([dimensions,1])
# portfolio_delta = inputs.delta_portfolio
# portfolio_beta = hedge.beta_portfolio_usd
# betas = hedge.betas
# regularization = 0.1
# optimal_result = minimize(fun= file_functions.cost_function_hedge, x0=x, args=(portfolio_delta, portfolio_beta, betas, regularization))
# optimal_hedge = optimal_result.x
# print(optimal_result)


