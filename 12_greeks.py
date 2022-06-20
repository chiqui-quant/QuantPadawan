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
inputs.volatility = 0.1442
inputs.interest_rate = 0.0158
inputs.maturity = 12/12 
inputs.strike = 20
inputs.call_or_put = 'call'

radius = 0.1 # Note: this allows you to 'zoom' the plot (change the scale)
inputs.price = np.linspace(1-radius, 1+radius,1000) * inputs.strike # to avoid zeros in the computation
months = [0,3,6,9,11] # vector of months for which we compute option prices
dict_plots = {}

# Compute option prices
for month in months:
    inputs.time = float(month) / (12 * inputs.maturity)
    price_black_scholes = file_functions.compute_price_black_scholes(inputs)
    dict_plots[month] = price_black_scholes

# Compute payoff
if inputs.call_or_put == 'call':
    payoff = np.array([max(s - inputs.strike, 0.0) for s in inputs.price])
elif inputs.call_or_put == 'put':
    payoff = np.array([max(inputs.strike - s, 0.0) for s in inputs.price])

# Plot option prices
plt.figure()
plt.title('Plot of option prices | ' + inputs.call_or_put + ' | Strike price: ' + str(inputs.strike) + '\n' + 'Volatility: ' + str(inputs.volatility) + ' | Interest rate: ' + str(inputs.interest_rate) )
for month in months:
    plt.plot(inputs.price, dict_plots[month], label=str(month) + '-month')
plt.plot(inputs.price, payoff, label = 'payoff')
plt.ylabel('Price option')
plt.xlabel('Price underlying')
plt.legend()
plt.grid()
plt.show()