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

def load_timeseries(ric):
    directory = 'C:\\Users\\Chiqui\\Desktop\\Python Projects\\QuantPadawan\\data\\'
    path = directory + ric + '.csv'
    raw_data = pd.read_csv(path)
    t = pd.DataFrame() # t (for table) with no elements
    t['date'] = pd.to_datetime(raw_data['Date'], dayfirst=True) # if dayfirst is omitted we have American dates
    t['close'] = raw_data['Close']
    t.sort_values(by='date',ascending=True)
    t['close_previous'] = t['close'].shift(1) # shift moves values below by the amount you indicate
    t['return_close'] = t['close']/t['close_previous'] - 1
    t = t.dropna()
    t = t.reset_index(drop=True)

    return t

def load_synchronized_timeseries(ric_x, ric_y):
    # Get timeseries of x and y
    table_x = file_functions.load_timeseries(ric_x)
    table_y = file_functions.load_timeseries(ric_y)
    # Note: we might have a different number of observations, we need to synchronize the timeseries
    # Synchronize timestamps
    timestamps_x = list(table_x['date'].values)
    timestamps_y = list(table_y['date'].values)
    timestamps = list(set(timestamps_x) & set(timestamps_y))
    # Sinchronized timeseries for x
    table_x_sync = table_x[table_x['date'].isin(timestamps)]
    table_x_sync.sort_values(by='date', ascending=True)
    table_x_sync = table_x_sync.reset_index(drop=True)
    # Sinchronized timeseries for y
    table_y_sync = table_y[table_y['date'].isin(timestamps)]
    table_y_sync.sort_values(by='date', ascending=True)
    table_y_sync = table_y_sync.reset_index(drop=True)
    # Table of returns for x and y
    t = pd.DataFrame()
    t['date'] = table_x_sync['date']
    t['price_x'] = table_x_sync['close']
    t['return_x'] = table_x_sync['return_close']
    t['price_y'] = table_y_sync['close']
    t['return_y'] = table_y_sync['return_close']

    return t

def cost_function_hedge(x, portfolio_delta, portfolio_beta, betas, regularization):

    n = len(x)
    deltas = np.ones([n])
    f_delta = (np.transpose(deltas).dot(x).item() + portfolio_delta)**2
    f_beta = (np.transpose(betas).dot(x).item() + portfolio_beta)**2
    penalty = regularization*(np.sum(x**2))
    f = f_delta + f_beta + penalty
    return f

def compute_portfolio_variance(x, covariance_matrix):
    variance = np.dot(x.T, np.dot(covariance_matrix, x)).item() # x(transpose)Vx
    return variance

def compute_portfolio_volatilty(x,covariance_matrix):
    notional = sum(abs(x))
    if notional <= 0.0:
        return 0.0
    variance = np.dot(x.T, np.dot(covariance_matrix, x)).item()
    if variance <= 0.0:
        return 0.0
    volatility = np.sqrt(variance)
    return volatility

def compute_efficient_frontier(rics, notional, target_return, include_min_var):
    # Special portfolios
    label1 = 'min-variance' # min-variance using eigenvectors
    label2 = 'pca' # pca using eigenvectors (max variance portfolio)
    label3 = 'equi-weight' #equi-weight
    label4 = 'volatility-weighted' # volatility-weighted
    label5 = 'long-only' # long-only portfolio with minimum variance
    label6 = 'markowitz-avg' # Markowitz with return = average of returns
    label7 = 'markowitz-target' # Markowitz with return =target_return

    # Compute covariance matrix
    port_mgr = file_classes.portfolio_manager(rics, notional)
    port_mgr.compute_covariance_matrix(bool_print=True)

    # Compute vectors of returns and volatilities for Markowitz portfolios
    min_returns = np.min(port_mgr.returns)
    max_returns = np.max(port_mgr.returns)
    returns = min_returns + np.linspace(0.1,0.9,100) + (max_returns-min_returns) # this way we generate 100 returns between min and max (in order to be inside the convex case we consider between 0.1 and 0.9)
    volatilities = np.zeros([len(returns),1])
    counter = 0
    for ret in returns: # compute markowitz portfolio for each return and compute its volatility
        port_markowitz = port_mgr.compute_portfolio('markowitz', ret)
        volatilities[counter] = port_markowitz.volatility_annual
        counter += 1

    # Compute special portfolios
    port1 = port_mgr.compute_portfolio(label1)
    port2 = port_mgr.compute_portfolio(label2)
    port3 = port_mgr.compute_portfolio(label3)
    port4 = port_mgr.compute_portfolio(label4)
    port5 = port_mgr.compute_portfolio(label5)
    port6 = port_mgr.compute_portfolio('markowitz')
    port7 = port_mgr.compute_portfolio('markowitz', target_return)

    # Create scatterplot of portfolios (volatility vs return)
    x1 = port1.volatility_annual
    y1 = port1.return_annual
    x2 = port2.volatility_annual
    y2 = port2.return_annual
    x3 = port3.volatility_annual
    y3 = port3.return_annual
    x4 = port4.volatility_annual
    y4 = port4.return_annual
    x5 = port5.volatility_annual
    y5 = port5.return_annual
    x6 = port6.volatility_annual
    y6 = port6.return_annual
    x7 = port7.volatility_annual
    y7 = port7.return_annual

    # Plot efficient frontier
    plt.figure()
    plt.title('Efficient Frontier for a portfolio including ' + rics[0])
    plt.scatter(volatilities,returns)
    if include_min_var:
        plt.plot(x1, y1, "ok", label=label1) # black cross
    plt.plot(x2, y2, "^r", label=label2) # red dot
    plt.plot(x3, y3, "^y", label=label3) # yellow square
    plt.plot(x4, y4, "^k", label=label4) # black square
    plt.plot(x5, y5, "sy", label=label5) # yellow triangle
    plt.plot(x6, y6, "sr", label=label6) # red square
    plt.plot(x7, y7, "sk", label=label7) # 
    plt.ylabel('portfolio return')
    plt.xlabel('portfolio volatility')
    plt.grid()
    if include_min_var:
        plt.legend(loc='best')
    else:
        plt.legend(loc='upper right', borderaxespad=0.)
    plt.show()

    dict_portfolios = {label1: port1,
                        label2: port2,
                        label3: port3,
                        label4: port4,
                        label5: port5,
                        label6: port6,
                        label7: port7,}
    return dict_portfolios
  
