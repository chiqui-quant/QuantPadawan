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

# Inputs
inputs = file_classes.hedge_input()
inputs.benchmark = 'IMIB.MI'
inputs.security = 'ISP.MI'
inputs.hedge_securities = ['EXO.MI', 'ENEL.MI']
inputs.delta_portfolio = 10 # mln EUR

# Compute hedges
hedge = file_classes.hedge_manager(inputs)
hedge.load_betas() # get the betas for portfolio and hedges
hedge.compute() # compute and print optimal hedge via CAPM

###################################
# original script to check solution
###################################

# Input parameters
# benchmark = 'IMIB.MI'
# security = 'A2A.MI' # or ric, Reuters Instruent Code (ticker-like code used by Refinitiv to identify financial instruments and indices)
# hedge_rics = ['AMP.MI', 'ATL.MI']
# delta_portfolio = 10 # mln EUR

# # Compute betas
# capm = file_classes.capm_manager(benchmark, security)
# capm.load_timeseries()
# capm.compute()
# beta_portfolio = capm.beta
# beta_portfolio_usd = beta_portfolio * delta_portfolio # mln EUR

# # Print input
# print('------')
# print('Input portfolio:')
# print('Delta mln EUR for ' + security + ' is ' + str(delta_portfolio))
# print('Beta for ' + security + ' vs ' + benchmark + ' is ' + str(beta_portfolio))
# print('Beta mln EUR for ' + security + ' vs ' + benchmark + ' is ' +  str(beta_portfolio_usd))

# # Compute betas for the hedges (construct an array of zeros and add the computed betas)
# shape = [len(hedge_rics),1]
# betas = np.zeros(shape)
# counter = 0
# print('------')
# print('Input hedges:')
# for hedge_ric in hedge_rics: 
#     capm = file_classes.capm_manager(benchmark, hedge_ric)
#     capm.load_timeseries()
#     capm.compute()
#     beta = capm.beta
#     print('Beta for hedge[' + str(counter) + '] = ' + hedge_ric + ' vs ' + benchmark + ' is ' + str(beta))
#     betas[counter] = beta
#     counter += 1

# # Exact solution using matrix algebra
# deltas = np.ones(shape) # vertical vector of ones
# targets = -np.array([[delta_portfolio],[beta_portfolio_usd]]) # our targets in order to hedge are -delta and -beta
# mtx = np.transpose(np.column_stack((deltas,betas))) # stack deltas and betas and take the transpose
# optimal_hedge = np.linalg.inv(mtx).dot(targets) # invert the matrix and multiply by targets
# hedge_delta = np.sum(optimal_hedge)
# hedge_beta_usd = np.transpose(betas).dot(optimal_hedge).item()

# # Print result
# print('------')
# print('Optimization result')
# print('------')
# print('Delta: ' + str(delta_portfolio))
# print('Beta USD: ' + str(beta_portfolio_usd))
# print('------')
# print('Hedge delta: ' + str(hedge_delta))
# print('Hedge beta: ' + str(hedge_beta_usd))
# print('------')
# print('Betas for the hedge: ')
# print(betas)
# print('------')
# print('Optimal hedge: ')
# print(optimal_hedge)
# print('------')

