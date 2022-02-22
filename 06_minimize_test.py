import pandas as pd
import numpy as np
import matplotlib as mpl 
import scipy
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2, linregress
from scipy.optimize import minimize

# # Example 1 (to verify how it works)
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

# # Example 2 (more general, generating random vectors anc checking solutions)
# # Define the function to minimize
# def cost_function(x, roots, coeffs):
#     f = 0
#     for n in range(len(x)):
#         f += coeffs[n]*(x[n] - roots[n])**2
#     return f
# # Input parameters (random vector of inputs)
# dimensions = 5 # dimension of input vector
# roots = np.random.randint(low=-20, high=20, size=5) # roots between -20 and 20
# coeffs = np.ones([dimensions,1])
# # Initialize optimization
# x = np.zeros([dimensions,1])
# # Compute optimization
# optimal_result = minimize(fun=cost_function, x0=x, args=(roots,coeffs))
# # Print
# print('------')
# print('Optimization result:')
# print(optimal_result)
# print('Roots:')
# print(roots)
# print('------')




