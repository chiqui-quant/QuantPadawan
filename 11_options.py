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
inputs = file_classes.option_input()
inputs.price = 102
inputs.time = 0.0 # in years
inputs.volatility = 0.25
inputs.interest_rate = 0.01
inputs.maturity = 2/12 # in years
inputs.strike = 100
inputs.call_or_put = 'call'
number_simulations = 1*10**6

# Price using Black-Scholes formula
price_black_scholes = file_functions.compute_price_black_scholes(inputs)
print('Price using Black-Scholes formula | ' + str(inputs.call_or_put) + '\n'\
    + str(price_black_scholes))
print('------')

# Price using Monte Carlo simulations
prices_monte_carlo = file_functions.compute_price_monte_carlo(inputs, number_simulations)
print(prices_monte_carlo)
prices_monte_carlo.plot_histogram()