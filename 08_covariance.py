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

# Note: small changes in inputs can give rise to large changes in the portfolio (variance-covariance matrix instability)
nb_decimals = 6 # 3 4 5 6
scale = 252 # 1 252 (for annualized covariance assuming brownian motion, so that variance increases with sqrt of time)
notional = 10 # mln EUR
rics = ['A2A.MI', 'AMP.MI', 'ATL.MI', 'AZM.MI', 'IP.MI']

# Compute covariance matrix via np.cov
# returns = [] # we create an empty list of returns and then append them (where each row is an array of a given asset returns)
# for ric in rics:
#     t = file_functions.load_timeseries()
#     x = t['return_close'].values
#     returns.append(x)
# mtx_covar = np.cov(returns) # cov = covariance
# mtx_correl = np.corrcoef(returns) # corrcoef = correlation
# # Note: this didn't work because the variance-covariance matrix needs to be squared, thus we need to synchronize timeseries first

"""
Important remark: of all the arrays we have we could synchronize all them and stay with the one with 
lower size. But when we reduce it this way not necessarily they will share the same timestamps. Imagine you 
reduce the highest number of timestamps with the lowest and you are using the STOXX600, suppose that unluckly
there is a null day in each day of the stocks in the STOXX600 but that null day is different
for each of the stocks, that would mean we would have to remove 600 days in the dataset (since the 
intersection of null days is empty). Now, we initially have around 1200 days, that would mean we would end 
up with 50% less of the original data information.

Solution: we consider and synchronize two securities at the time and compute the covariance and
the correlation between them. This way we will end up with more information.
"""

# Compute variance-covariance matrix by pairwise covariances
size = len(rics) # Number of securities we are playing with
mtx_covar = np.zeros([size,size]) # create an NxN matrix for covariances
mtx_correl = np.zeros([size,size]) # create an NxN matrix for correlations
vec_returns = np.zeros([size,1]) # compute vertical vector of annualized returns (element 1 = annualized return of security 1, element 2 = annualized return of security 2...)
vec_volatilities = np.zeros([size,1]) # same for volatilities (eg. element 5 = annualized volatility of asset 5)
returns = []
for i in range(size):
    ric_x = rics[i]
    for j in range(i+1):
        ric_y = rics[j]
        t = file_functions.load_synchronized_timeseries(ric_x, ric_y)
        ret_x = t['return_x'].values
        ret_y = t['return_y'].values
        returns = [ret_x, ret_y] # now we have an array like list of synchronize returns and we can compute cov and corr matrices
        # Covariances
        temp_mtx = np.cov(returns)
        temp_covar = scale*temp_mtx[0][1] # we compute the annualized covariances
        temp_covar = np.round(temp_covar, nb_decimals) # we round them
        mtx_covar[i][j] = temp_covar
        mtx_covar[j][i] = temp_covar
        # Correlations (same process)
        temp_mtx = np.corrcoef(returns)
        temp_correl = temp_mtx[0][1] 
        temp_correl = np.round(temp_correl, nb_decimals) 
        mtx_correl[i][j] = temp_correl
        mtx_correl[j][i] = temp_correl
        if j == 0: # if j = 0 update reuturns (because we don't want to update returs constantly all the time, only once)
            temp_ret = ret_x
        # Mean returns
        temp_mean = np.round(scale*np.mean(temp_ret), nb_decimals) # return is linear in time (brownian motion)
        vec_returns[i] = temp_mean
        # Volatilities
        temp_volatility = np.round(np.sqrt(scale)*np.std(temp_ret), nb_decimals) # volatility grows with the square root of time
        vec_volatilities[i] = temp_volatility

"""Note: using range we get n-1 (eg. range 5, we get 1 to 4) this is why we use range(i+1) for j
because once we have i, we want j to be the same number as i. Eg. i(row)=3 and j(column)=3. If we
did i until range(size) and j until range(size) we would be computing 2 times the same element since the 
matrix is symmetric (useless additional computation)""" 
# for i in range(5): 
#     print(i)

# Now we can compute the eigenvalues and eigenvectors in 2 ways
# Compute eigenvalues and eigenvectors 
# eigenvalues, eigenvectors = LA.eig(mtx_covar)
# Compute eigenvalues and eigenvectors for symmetric matrices
eigenvalues, eigenvectors = LA.eigh(mtx_covar)
# Note: in the second way (for symmetric matrices) the eigenvalues and eigenvectors will be normalized (ordered), which is what we want
# Note: the vertical eigenvector of index 0 is the minimum variance portfolio (it is the eigenvector associated with the lowest eigenvalue)

print('------')
print('Securities:')
print(rics)
print('------')
print('Returns (annualized):')
print(vec_returns)
print('------')
print('Volatilities (annualized):')
print(vec_volatilities)
print('------')
print('Variance-covariance matrix (annualized):')
print(mtx_covar)
print('------')
print('Correlation matrix (annualized):')
print(mtx_correl)
print('------')
print('Eigenvalues:')
print(eigenvalues)
print('------')
print('Eigenvectors:')
print(eigenvectors)

# Min-variance portfolio
print('------')
print('Min-variance portfolio:')
print('notional (mln EUR) = ' + str(notional))
variance_explained = eigenvalues[0] / sum(abs(eigenvalues)) # R^2 of the variance explained by the eigenvector (abs = absolute value)
eigenvector = eigenvectors[:,0] # first column is the min-variance eigenvector
port_min_var = notional * eigenvector / sum(abs(eigenvector))
delta_min_var = sum(port_min_var)
print('delta (mln EUR) = ' + str(delta_min_var))
print('variance explained = ' + str(variance_explained))

# PCA (max-variance) portfolio
print('------')
print('PCA portfolio (max-variance):')
print('notional (mln EUR) = ' + str(notional))
variance_explained = eigenvalues[-1] / sum(abs(eigenvalues)) # R^2 of the variance explained by the eigenvector
eigenvector = eigenvectors[:,-1] # first column is the min-variance eigenvector
port_pca = notional * eigenvector / sum(abs(eigenvector))
delta_pca = sum(port_pca)
print('delta (mln EUR) = ' + str(delta_pca))
print('variance explained = ' + str(variance_explained))
# Note: the only difference for the max-variance portfolio is that we take the last column



