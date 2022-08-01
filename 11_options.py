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
inputs.price = 10.97
inputs.time = 0.0 # in years (starting time)
inputs.volatility = 0.29182
inputs.interest_rate = 0.005 # 0.01 = 1% interest rate
inputs.maturity = 5/12 # in years (maturity)
inputs.strike = 13
inputs.call_or_put = 'call'
number_simulations = 1*10**6

# Print Inputs
file_classes.option_input.print_inputs(inputs)

# Price using Black-Scholes formula
price_black_scholes = file_functions.compute_price_black_scholes(inputs)
print('Price using Black-Scholes formula | ' + str(inputs.call_or_put) + '\n'\
    + str(price_black_scholes))
print('------')

# Price using Monte Carlo simulations
prices_monte_carlo = file_functions.compute_price_monte_carlo(inputs, number_simulations)
print(prices_monte_carlo)
prices_monte_carlo.plot_histogram()