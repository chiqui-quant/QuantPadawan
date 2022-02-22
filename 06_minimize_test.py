import pandas as pd
import numpy as np
import matplotlib as mpl 
import scipy
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2, linregress
from scipy.optimize import minimize

# # Example 1
# # Define the function to minimize
# def cost_function(x):
#     f = (x[0] - 7.0)**2 + (x[1] + 5)**2 + (x[2] - 13)**2
#     return f
# # Initialize optimization
# x = np.zeros([3,1])
# # Compute optimization
# optimal_result = scipy.optimize.minimize(fun=cost_function, x0=x)
# # Print
# print('------')
# print('Optimization result:')
# print(optimal_result)

 