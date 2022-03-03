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

price_black_scholes = file_functions.compute_price_black_scholes(inputs)
print(price_black_scholes)