import pandas as pd
import numpy as np
import matplotlib as mpl 
import scipy
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2, linregress
from scipy.optimize import minimize
from numpy import linalg as LA, size

# Import our own files and reload
import file_classes
importlib.reload(file_classes)
import file_functions
importlib.reload(file_functions)


class distribution_manager():
    
    def __init__(self, inputs):
        self.inputs = inputs # distribution inputs
        self.data_table = None
        self.description = None
        self.nb_rows = None
        self.mean = None
        self.std = None
        self.skew = None
        self.kurtosis = None # excess kurtosis
        self.jb_stat = None
        self.p_value = None
        self.is_normal = None 
        self.sharpe = None
        self.var_95 = None
        self.cvar_95 = None
        self.percentile_25 = None
        self.median = None
        self.percentile_75 = None

    """ Note: these are all the attributes we have access to, even if the file could run without them
    it is a good convention to report them so we know it immediately (eg. run dm.mean or any other metric)
    """

    def __str__(self):
        str_self = self.description + ' | size ' + str(self.nb_rows) + '\n' + self.plot_str()
        return str_self

    def load_timeseries(self):
        
        data_type = self.inputs.data_type

        if data_type == 'simulation':
            nb_sims = self.inputs.nb_sims
            dist_name = self.inputs.variable_name
            degrees_freedom = self.inputs.degrees_freedom
            if dist_name == 'normal':
                x = np.random.standard_normal(nb_sims)
                x_description = data_type + ' ' + dist_name
            elif dist_name == 'exponential':
                x = np.random.standard_exponential(nb_sims)
                x_description = data_type + ' ' + dist_name
            elif dist_name == 'uniform':
                x = np.random.uniform(0,1,nb_sims)
                x_description = data_type + ' ' + dist_name
            elif dist_name == 'student':
                x = np.random.standard_t(df=degrees_freedom,size=nb_sims)
                x_description = data_type + ' ' + dist_name + ' | df = ' + str(degrees_freedom)
            elif dist_name == 'chi-square':
                x = np.random.chisquare(df=degrees_freedom,size=nb_sims)
                x_description = data_type + ' ' + dist_name + ' | df = ' + str(degrees_freedom)
                
                self.description = x_description
                self.nb_rows = nb_sims
                self.vec_returns = x


        elif data_type == 'real':
            directory = 'C:\\Users\\Chiqui\\Desktop\\Python Projects\\QuantPadawan\\data\\'

            ric = self.inputs.variable_name
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
            
            self.data_table = t
            self.description = 'market data ' + ric
            self.nb_rows = t.shape[0]
            self.vec_returns = t['return_close'].values #

            ric = self.inputs.variable_name
            t = file_functions.load_timeseries(ric)

    def plot_histogram(self):
        plt.figure()
        plt.hist(self.vec_returns,bins=100) # bins is the interval of the rectangles in the x axis
        plt.title(self.description)
        plt.xlabel(self.plot_str())
        plt.show()

    def compute(self):
        
        self.mean = np.mean(self.vec_returns)
        self.std = np.std(self.vec_returns)
        self.skew = skew(self.vec_returns)
        self.kurtosis = kurtosis(self.vec_returns) # excess kurtosis
        self.jb_stat = self.nb_rows/6*(self.skew**2 + 1/4*self.kurtosis**2)
        self.p_value = 1 - chi2.cdf(self.jb_stat, df=2) 
        self.is_normal = (self.p_value > 0.05) # equivalently jb < 6
        self.sharpe = self.mean / self.std * np.sqrt(252) # annualized Sharpe ratio under the hypothesis that standard deviation grows as sqrt(time) increases
        self.var_95 = np.percentile(self.vec_returns,5)
        self.cvar_95 = np.mean(self.vec_returns[self.vec_returns <= self.var_95]) # mean of returns on the left of var_95
        self.percentile_25 = self.percentile(25) # alternatively np.percentile(self.vec_returns,25)
        self.median = np.median(self.vec_returns)
        self.percentile_75 = self.percentile(75)

    def plot_str(self):
        nb_decimals = 4
        plot_str = 'mean ' + str(np.round(self.mean, nb_decimals))\
            + ' | std dev ' + str(np.round(self.std, nb_decimals))\
            + ' | skewness ' + str(np.round(self.skew, nb_decimals))\
            + ' | kurtosis ' + str(np.round(self.kurtosis, nb_decimals)) + '\n'\
            + 'Jarque Bera  ' + str(np.round(self.jb_stat, nb_decimals))\
            + ' | p-value ' + str(np.round(self.p_value, nb_decimals))\
            + ' | is normal ' + str(self.is_normal) + '\n'\
            + 'Sharpe annual ' + str(np.round(self.sharpe, nb_decimals))  + ' | VaR 95% ' + str(np.round(self.var_95, nb_decimals))\
            + ' | CVaR 95% ' + str(np.round(self.cvar_95, nb_decimals)) + '\n'\
            + 'percentile 25% ' + str(np.round(self.percentile_25, nb_decimals))\
            + ' | median ' + str(np.round(self.median, nb_decimals))\
            + ' | percentile 75% ' + str(np.round(self.percentile_75, nb_decimals))
        return plot_str

    def percentile(self, pct):
        percentile = np.percentile(self.vec_returns, pct)
        return percentile

class distribution_input():
    
    def __init__(self):
        self.data_type = None # simulation real custom
        self.variable_name = None # normal student exponential chi-square uniform 'TICKER'
        self.degrees_freedom = None # only used in simulation + student and chi-square
        self.nb_sims = None # only in simulation

class capm_manager():
    
    def __init__(self, benchmark, security):
        self.benchmark = benchmark
        self.security = security
        self.nb_decimals = 4
        self.data_table = None
        self.alpha = None
        self.beta = None
        self.p_value = None
        self.correlation = None
        self.r_squared = None
        self.std_err = None
        self.predictor_linreg = None

    def __str__(self):
        return self.plot_str()

    def plot_str(self):
        str_self = 'Linear regression | security ' + self.security\
            + ' | benchmark ' + self.benchmark + '\n'\
            + 'alpha (intercept) ' + str(self.alpha)\
            + ' | beta (slope) ' + str(self.beta) + '\n'\
            + 'p_value ' + str(self.p_value)\
            + ' | null hypothesis ' + str(self.null_hypothesis) + '\n'\
            + 'correl (r-value) ' + str(self.correlation)\
            + ' | r-squared ' + str(self.r_squared)
        return str_self


    def load_timeseries(self):
        self.data_table = file_functions.load_synchronized_timeseries(ric_x=self.benchmark, ric_y=self.security)
    
    def compute(self):
        # Linear regression
        x = self.data_table['return_x'].values
        y = self.data_table['return_y'].values
        slope, intercept, r_value, p_value, std_err = linregress(x,y)
        self.alpha = np.round(intercept, self.nb_decimals)
        self.beta = np.round(slope, self.nb_decimals)
        self.p_value = np.round(p_value, self.nb_decimals)
        self.null_hypothesis = p_value > 0.05 # if p_value < 0.05 reject the null hypothesis (that is, that the linear regression is bad, eg. ^STOXX50E with EURUSD=X)
        self.correlation = np.round(r_value, self.nb_decimals) # correlation coefficient
        self.r_squared = np.round(r_value**2, self.nb_decimals) # pct of variance of y explained by x
        self.predictor_linreg = intercept + slope*x

    def plot_timeseries(self):
        plt.figure(figsize=(12,5))
        plt.title('Time series of price')
        plt.xlabel('Time')
        plt.ylabel('Prices')
        ax = plt.gca()
        ax1 = self.data_table.plot(kind='line', y='price_x', ax=ax, grid=True,\
            color='blue', label=self.benchmark)
        ax2 = self.data_table.plot(kind='line', y='price_y', ax=ax, grid=True,\
            color='red',secondary_y=True, label=self.security)
        ax1.legend(loc=2)
        ax2.legend(loc=1)
        plt.show() 

    
    def plot_linear_regression(self):
        x = self.data_table['return_x'].values
        y = self.data_table['return_y'].values
        str_title = 'Scatterplot of returns' + '\n' + self.plot_str()
        plt.figure() 
        plt.title(str_title)
        plt.scatter(x,y)
        plt.plot(x, self.predictor_linreg, color='green')
        plt.xlabel(self.benchmark)
        plt.ylabel(self.security)
        plt.grid()
        plt.show()   

class hedge_manager():
    
    def __init__(self, inputs):
        self.inputs = inputs # hedge inputs
        self.benchmark = inputs.benchmark # the market in CAPM (in general a market index)
        self.security = inputs.security # portfolio to hedge
        self.hedge_securities = inputs.hedge_securities # hedge universe
        self.nb_hedges = len(self.hedge_securities)
        self.delta_portfolio = inputs.delta_portfolio
        self.beta_portfolio = None
        self.beta_portfolio_usd = None
        self.betas = None
        self.optimal_hedge = None 
        self.hedge_delta = None
        self.hedge_beta_usd = None

    def load_betas(self):
        benchmark = self.benchmark
        security = self.security
        hedge_securities = self.hedge_securities
        delta_portfolio = self.delta_portfolio
        # Compute betas for the portfolio
        capm = file_classes.capm_manager(benchmark, security)
        capm.load_timeseries()
        capm.compute()
        beta_portfolio = capm.beta
        beta_portfolio_usd = beta_portfolio * delta_portfolio # mln EUR
        # Print input
        print('------')
        print('Input portfolio:')
        print('Delta mlnEUR for ' + security + ' is ' + str(delta_portfolio))
        print('Beta for ' + security + ' vs ' + benchmark + ' is ' + str(beta_portfolio))
        print('Beta mlnEUR for ' + security + ' vs ' + benchmark + ' is ' +  str(beta_portfolio_usd))
        # Compute betas for the hedges (construct an array of zeros and add the computed betas)
        shape = [len(hedge_securities),1]
        betas = np.zeros(shape)
        counter = 0
        print('------')
        print('Input hedges:')
        for hedge_ric in hedge_securities: 
            capm = file_classes.capm_manager(benchmark, hedge_ric)
            capm.load_timeseries()
            capm.compute()
            beta = capm.beta
            print('Beta for hedge[' + str(counter) + '] = ' + hedge_ric + ' vs ' + benchmark + ' is ' + str(beta))
            betas[counter] = beta
            counter += 1

        self.beta_portfolio = beta_portfolio
        self.beta_portfolio_usd = beta_portfolio_usd
        self.betas = betas

    def compute(self, regularization=0.0):
        # Numerical solution
        dimensions = len(self.hedge_securities)
        x = np.zeros([dimensions,1])
        betas = self.betas
        optimal_result = minimize(fun= file_functions.cost_function_hedge, x0=x, args=(self.delta_portfolio, self.beta_portfolio_usd, self.betas, regularization))
        self.optimal_hedge = optimal_result.x
        # Compute the delta and beta of hedge portfolio
        self.hedge_delta = np.sum(self.optimal_hedge)
        self.hedge_beta_usd = np.transpose(betas).dot(self.optimal_hedge).item()
        # Print result
        self.print_result('numerical')

    def compute_exact(self):
        # Exact solution using matrix algebra
        n = len(self.hedge_securities)
        if n != 2: 
            print('------')
            print('Cannot compute exact solution because n = ' + str(n) + ' =/= 2')
            return
        shape = [n]
        betas = self.betas
        deltas = np.ones(shape) # vertical vector of ones
        targets = -np.array([[self.delta_portfolio],[self.beta_portfolio_usd]]) # our targets in order to hedge are -delta and -beta
        mtx = np.transpose(np.column_stack((deltas,betas))) # stack deltas and betas and take the transpose
        self.optimal_hedge = np.linalg.inv(mtx).dot(targets) # invert the matrix and multiply by targets
        # Compute the delta and beta of hedge portfolio
        self.hedge_delta = np.sum(self.optimal_hedge)
        self.hedge_beta_usd = np.transpose(betas).dot(self.optimal_hedge).item()
        # Print result
        self.print_result('exact')

    def print_result(self, algo_type):
        # Print result
        print('------')
        print('Optimization result - ' + algo_type + ' solution')
        print('------')
        print('Delta portfolio: ' + str(self.delta_portfolio))
        print('Beta portfolio EUR: ' + str(self.beta_portfolio_usd))
        print('------')
        print('Delta hedge: ' + str(self.hedge_delta))
        print('Beta hedge EUR: ' + str(self.hedge_beta_usd))
        print('------')
        print('Optimal hedge: ')
        print(self.optimal_hedge)


class hedge_input():

    def __init__(self):
        self.benchmark = None # the market in CAPM (in general ^STOXX50E)
        self.security = 'A2A.MI' # portfolio to hedge
        self.hedge_securities = ['AMP.MI', 'ATL.MI'] # hedge universe
        self.delta_portfolio = 10 # mlnEUR (default 10)

class portfolio_manager():

    def __init__(self, rics, notional):
        self.rics = rics
        self.size = len(rics)
        self.notional = notional
        self.nb_decimals = 6
        self.scale = 252
        self.covariance_matrix = None
        self.correlation_matrix = None
        self.returns = None
        self.volatilities = None

    def compute_covariance_matrix(self, bool_print=True):
        rics = self.rics
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
                temp_covar = self.scale*temp_mtx[0][1] # we compute the annualized covariances
                temp_covar = np.round(temp_covar, self.nb_decimals) # we round them
                mtx_covar[i][j] = temp_covar
                mtx_covar[j][i] = temp_covar
                # Correlations (same process)
                temp_mtx = np.corrcoef(returns)
                temp_correl = temp_mtx[0][1]
                temp_correl = np.round(temp_correl, self.nb_decimals) 
                mtx_correl[i][j] = temp_correl
                mtx_correl[j][i] = temp_correl
                if j == 0: # if j = 0 update returns (because we don't want to update returs constantly all the time, only once)
                    temp_ret = ret_x
                # Mean returns
                temp_mean = np.round(self.scale*np.mean(temp_ret), self.nb_decimals) # return is linear in time (brownian motion)
                vec_returns[i] = temp_mean
                # Volatilities
                temp_volatility = np.round(np.sqrt(self.scale)*np.std(temp_ret), self.nb_decimals) # volatility grows with the square root of time
                vec_volatilities[i] = temp_volatility
        # Compute eigenvalues and eigenvectors for symmetric matrices
        eigenvalues, eigenvectors = LA.eigh(mtx_covar)

        self.covariance_matrix = mtx_covar
        self.correlation_matrix = mtx_correl
        self.returns = vec_returns
        self.volatilities = vec_volatilities
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors

        if bool_print:
            print('------')
            print('Securities:')
            print(self.rics)
            print('------')
            print('Returns (annualized):')
            print(self.returns)
            print('------')
            print('Volatilities (annualized):')
            print(self.volatilities)
            print('------')
            print('Variance-covariance matrix (annualized):')
            print(self.covariance_matrix)
            print('------')
            print('Correlation matrix (annualized):')
            print(self.correlation_matrix)
            # print('------')
            # print('Eigenvalues:')
            # print(self.eigenvalues)
            # print('------')
            # print('Eigenvectors:')
            # print(self.eigenvectors)

    def compute_portfolio(self, portfolio_type='default', target_return=None):
        
        portfolio = portfolio_item(self.rics, self.notional)
        
        if portfolio_type == 'min-variance':
            portfolio.type = portfolio_type
            portfolio.variance_explained = self.eigenvalues[0] / sum(abs(self.eigenvalues)) # R^2 of the variance explained by the eigenvector (abs = absolute value)
            eigenvector = self.eigenvectors[:,0] # first column is the min-variance eigenvector
            if max(eigenvector) < 0: # if all weights aree negative, return a long-only portfolio
                eigenvector = - eigenvector
            weights_normalized = eigenvector / sum(abs(eigenvector))

        elif portfolio_type == 'min-variance-l1':
            portfolio.type = portfolio_type
            # Initialize optimization
            x = np.zeros([self.size,1])
            # Initialize constraints
            cons = [{"type": "eq", "fun": lambda x: sum(abs(x)) - 1}] # unitary in norm L1
            # Compute optimization
            res = minimize(file_functions.compute_portfolio_variance, x, args=(self.covariance_matrix), constraints=cons)
            weights_normalized = res.x

        elif portfolio_type == 'min-variance-l2':
            portfolio.type = portfolio_type
            # Initialize optimization
            x = np.zeros([self.size,1])
            # Initialize constraints
            cons = [{"type": "eq", "fun": lambda x: sum(x**2) - 1}] # unitary in norm L2
            # Compute optimization
            res = minimize(file_functions.compute_portfolio_variance, x, args=(self.covariance_matrix), constraints=cons)
            weights_normalized = res.x / sum(abs(res.x))
        
        elif portfolio_type == 'pca':
            portfolio.type = portfolio_type
            portfolio.variance_explained = self.eigenvalues[-1] / sum(abs(self.eigenvalues)) # R^2 of the variance explained by the eigenvector (abs = absolute value)
            eigenvector = self.eigenvectors[:,-1] # first column is the min-variance eigenvector
            if max(eigenvector) < 0:
                eigenvector = - eigenvector
            weights_normalized = eigenvector / sum(abs(eigenvector))

        elif portfolio_type == 'long-only':
            portfolio.type = portfolio_type
            # Initialize optimization
            x = np.zeros([self.size,1])
            # Initialize constraints
            cons = [{"type": "eq", "fun": lambda x: np.sum(abs(x)) - 1}] 
            bnds = [(0, None) for i in range(self.size)]
            # Compute optimization
            res = minimize(file_functions.compute_portfolio_variance, x, args=(self.covariance_matrix), constraints=cons, bounds=bnds)
            weights_normalized = res.x

        elif portfolio_type == 'default' or portfolio_type == 'equi-weight':
            portfolio.type = 'equi-weight'
            weights_normalized = (1 / self.size) * np.ones([self.size])

        elif portfolio_type == 'markowitz':
            portfolio.type = portfolio_type
            if target_return == None:
                target_return = np.mean(self.returns)
            portfolio.target_return = target_return
            # Initialize optimization
            x = np.zeros([self.size,1])
            # Initialize constraints
            cons = [{"type": "eq", "fun": lambda x: np.transpose(self.returns).dot(x).item() - target_return}, \
                {"type": "eq", "fun": lambda x: np.sum(abs(x)) - 1}] # dictionary list of the constraints
            bnds = [(0, None) for i in range(self.size)]
            # Compute optimization
            res = minimize(file_functions.compute_portfolio_variance, x, args=(self.covariance_matrix), constraints=cons, bounds=bnds)
            weights_normalized = res.x

        elif portfolio_type == 'volatility-weighted': # more weight to lower volatility securities
            portfolio.type = portfolio_type
            x = 1 / self.volatilities
            weights_normalized = 1 / np.sum(x) * x

        weights = self.notional * weights_normalized

        portfolio.weights = weights
        portfolio.notional = sum(abs(weights))
        portfolio.delta = sum(weights)
        portfolio.pnl_annual_usd = np.transpose(self.returns).dot(weights).item()
        portfolio.return_annual = np.transpose(self.returns).dot(weights_normalized).item()
        portfolio.volatility_annual = file_functions.compute_portfolio_volatilty(weights_normalized, self.covariance_matrix)
        portfolio.volatility_annual_usd = file_functions.compute_portfolio_volatilty(weights, self.covariance_matrix)
        try:
            portfolio.sharpe_annual = portfolio.return_annual / portfolio.volatility_annual
        except ZeroDivisionError:
                portfolio.sharpe_annual = 0
        return portfolio


class portfolio_item():

    def __init__(self, rics, notional):
        self.rics = rics
        self.notional = notional
        self.type = ''
        self.weights = []
        self.delta = 0.0
        self.variance_explained = None
        self.pnl_annual_usd = None
        self.volatility_annual_usd = None
        self.target_return = None
        self.return_annual = None
        self.volatility_annual = None
        self.sharpe_annual = None

    def summary(self):
        print('------')
        print('Portfolio type: ' + self.type)
        print('Rics:')
        print(self.rics)
        print('Weights:')
        print(self.weights)
        print('Notional (mln EUR): ' + str(self.notional))
        print('Delta (mln EUR): ' + str(self.delta))
        if not self.variance_explained == None:
            print('Variance explained: ' + str(self.variance_explained))
        if not self.pnl_annual_usd == None:
            print('Profit and loss annual (mln EUR): ' + str(self.pnl_annual_usd))
        if not self.volatility_annual_usd == None:
            print('Volatility annual (mln EUR): ' + str(self.volatility_annual_usd))
        if not self.target_return == None:
            print('Target return: ' + str(self.target_return))
        if not self.return_annual == None:
            print('Return annual: ' + str(self.return_annual))
        if not self.volatility_annual == None:
            print('Volatility annual: ' + str(self.volatility_annual))
        if not self.sharpe_annual == None:
            print('Sharpe ratio annual: ' + str(self.sharpe_annual))
               
class option_input:

    def __init__(self):
        self.price = None                
        self.time = None                
        self.volatility = None                
        self.interest_rate = None       
        self.maturity = None  
        self.strike = None
        self.call_or_put = None 

class montecarlo_item():

    def __init__(self, sim_prices, sim_payoffs, strike, call_or_put):
        self.number_simulations = len(sim_payoffs)
        self.sim_prices = sim_prices
        self.sim_payoffs = sim_payoffs
        self.call_or_put = call_or_put
        self.mean = np.mean(sim_payoffs)
        self.std = np.std(sim_payoffs)
        self.confidence_radius = 1.96*self.std/np.sqrt(self.number_simulations)
        self.confidence_interval =  self.mean + np.array([-1,1])*self.confidence_radius
        if call_or_put == 'call':
            self.proba_exercise = np.mean(sim_prices > strike)
        elif call_or_put == 'put':
            self.proba_exercise = np.mean(sim_prices < strike)
        self.proba_profit = np.mean(sim_payoffs > self.mean)
        
        
    def __str__(self):
        str_self = 'Monte Carlo simulation for option pricing | ' + self.call_or_put + '\n'\
            + 'number of simulations ' + str(self.number_simulations) + '\n'\
            + 'confidence radius ' + str(self.confidence_radius) + '\n'\
            + 'confidence interval ' + str(self.confidence_interval) + '\n'\
            + 'price ' + str(self.mean)  + '\n'\
            + 'probability of exercise ' + str(self.proba_exercise)  + '\n'\
            + 'probability of profit ' + str(self.proba_profit)  
                
        return str_self
    
    
    def plot_histogram(self):
        inputs_distribution = file_classes.distribution_input()
        dm = file_classes.distribution_manager(inputs_distribution)
        dm.description = 'Monte Carlo distribution | option price | ' + self.call_or_put
        dm.nb_rows = len(self.sim_payoffs)
        dm.vec_returns = self.sim_payoffs
        dm.compute() # compute returns and all different risk metrics
        dm.plot_histogram() # plot histogram
        print(dm) # write all data in console

class backtest:
    
    def __init__(self):
        self.ric_long = 'TOTF.PA' # numerator
        self.ric_long = 'REP.MC' # denominator
        self.rolling_days = 20 # N
        self.level_1 = 1. # a
        self.level_2 = 2. # b
        self.data_cut = 0.7 # 70% in-sample and 30% out-of-sample
        self.data_type = 'in-sample' # in-sample out-of-sample
        self.dataframe_data = pd.DataFrame() # dataframe of data
        self.dataframe_indicators = pd.DataFrame() # dataframe of indicators
        self.dataframe_strategy = pd.DataFrame() # Note: cumulative dataframes (this contains the previous two)
        self.pnl_mean = np.NaN
        self.pnl_volatility = np.NaN
        self.pnl_sharpe = np.NaN
        self.nb_trades = 0


    def load_data(self):
        # Load data
        t = file_functions.load_synchronized_timeseries(self.ric_long, self.ric_short)
        cut = int(self.data_cut*t.shape[0])
        if self.data_type == 'in-sample':
            df = t.head(cut)
        elif self.data_type == 'out-of-sample':
            df = t.tail(t.shape[0]-cut)
        df = df.reset_index(drop=True)
        # Spread at current close
        df['spread'] = df['price_x']/df['price_y'] # Note: the stocks should be quoted in the same currency
        base = df['spread'][0] # normalize spread so we can compare different pairs of stocks
        df['spread'] = df['spread'] / base
        # Spread at previous close
        df['spread_previous'] = df['price_x_previous']/df['price_y_previous'] 
        df['spread_previous'] = df['spread_previous'] / base
        self.dataframe_data = df
        
    def compute_indicator(self,bool_plot=False):
        # Compute bollinger bands
        size = self.dataframe_data.shape[0]
        columns = ['lower_2','lower_1', 'mean', 'upper_1', 'upper_2']
        mtx_bollinger = np.empty((size, len(columns)))
        mtx_bollinger[:] = np.nan
        for n in range(self.rolling_days-1,size):
            vec_price = self.dataframe_data['spread'].values
            vec_price = vec_price[n-self.rolling_days+1:n+1] # to check
            mu = np.mean(vec_price)
            sigma = np.std(vec_price)
            m = 0
            mtx_bollinger[n][m] = mu - self.level_2*sigma
            m = m+1
            mtx_bollinger[n][m] = mu - self.level_1*sigma
            m = m+1
            mtx_bollinger[n][m] = mu 
            m = m+1
            mtx_bollinger[n][m] = mu + self.level_1*sigma
            m = m+1
            mtx_bollinger[n][m] = mu + self.level_2*sigma
            m = m+1
        df1 = pd.DataFrame(data=mtx_bollinger, columns=columns)
        df2 = pd.concat([self.dataframe_data,df1], axis=1) # axis=0 for rows, axis=1 for columns
        df2 = df2.dropna()
        df2 = df2.reset_index(drop=True)
        self.dataframe_indicators = df2
        if bool_plot:
            self.plot_indicators()
            
    def run_strategy(self, bool_plot=False):
        # Loop for backtest
        size = self.dataframe_indicators.shape[0]
        columns = ['position','entry_signal','exit_signal', 'pnl_daily', 'trade', 'pnl_trade']
        position = 0
        entry_spread = 0.
        can_trade = False
        mtx_backtest = np.zeros((size,len(columns)))
        for n in range(size):
            # input data for the day
            spread = self.dataframe_indicators['spread'][n]
            spread_previous = self.dataframe_indicators['spread_previous'][n]
            lower_2 = self.dataframe_indicators['lower_2'][n]
            lower_1 = self.dataframe_indicators['lower_1'][n]
            mean = self.dataframe_indicators['mean'][n]
            upper_1 = self.dataframe_indicators['upper_1'][n]
            upper_2 = self.dataframe_indicators['upper_2'][n]
            # reset output data for the day
            pnl_daily = 0.
            trade = 0
            pnl_trade = 0.
            # check if we can trade
            if not can_trade:
                can_trade = position == 0 and spread > lower_1 and spread < upper_1
            if not can_trade:
                continue
            # enter new position
            if position == 0:
                entry_signal = 0
                exit_signal = 0
                if spread > lower_2 and spread < lower_1:
                    entry_signal = 1 # buy signal
                    position = 1 
                    entry_spread = spread
                elif spread > upper_1 and spread < upper_2:
                    entry_signal = -1 # sell signal
                    position = -1
                    entry_spread = spread
            # exit long position
            elif position == 1:
                entry_signal = 0
                pnl_daily = position*(spread - spread_previous)
                if n == size-1 or spread > mean or spread < lower_2:
                    exit_signal = 1 # last day, take profit or stop loss
                    pnl_trade = position*(spread-entry_spread)
                    position = 0
                    trade = 1
                    can_trade = False
                else:
                    exit_signal = 0
            # exit short position
            elif position == -1:
                entry_signal = 0
                pnl_daily = position*(spread - spread_previous)
                if n == size-1 or spread < mean or spread > upper_2:
                    exit_signal = 1 # last day, take profit or stop loss
                    pnl_trade = position*(spread-entry_spread)
                    position = 0
                    trade = 1
                    can_trade = False
                else:
                    exit_signal = 0    

            # save data for the day
            m = 0
            mtx_backtest[n][m] = position
            m = m + 1 
            mtx_backtest[n][m] = entry_signal
            m = m + 1 
            mtx_backtest[n][m] = exit_signal
            m = m + 1 
            mtx_backtest[n][m] = pnl_daily
            m = m + 1 
            mtx_backtest[n][m] = trade
            m = m + 1 
            mtx_backtest[n][m] = pnl_trade
            
        df1 = pd.DataFrame(data=mtx_backtest,columns=columns)
        df2 = pd.concat([self.dataframe_indicators,df1],axis = 1) # axis=0 for rows, axis=1 for columns
        df2 = df2.dropna()
        df2 = df2.reset_index(drop=True)
        df2['cum_pnl_daily'] = np.cumsum(df2['pnl_daily'])
        self.dataframe_strategy = df2
        # Compute Sharpe ratio and number of trades
        vec_pnl = df2['pnl_daily'].values
        self.pnl_mean = np.round(np.mean(vec_pnl)* 252, 4) # annual PNL
        self.pnl_volatility = np.round(np.std(vec_pnl) * np.sqrt(252), 4) # annual PNL volatility
        self.pnl_sharpe = np.round(self.pnl_mean / self.pnl_volatility, 4)
        df3 = df2[df2['trade'] == 1]
        self.nb_trades = df3.shape[0]
        if bool_plot:
            self.plot_strategy()

        
    def plot_indicators(self):        
        timestamps = self.dataframe_indicators['date']
        spread = self.dataframe_indicators['spread']
        mu = self.dataframe_indicators['mean']
        u1 = self.dataframe_indicators['upper_1']
        u2 = self.dataframe_indicators['upper_2']
        l1 = self.dataframe_indicators['lower_1']
        l2 = self.dataframe_indicators['lower_2']
        plt.figure()
        plt.title('Spread ' + self.ric_long + ' / ' + self.ric_short)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.plot(timestamps, mu, color='blue', label='mean')
        plt.plot(timestamps, l1, color='green', label='lower_1')
        plt.plot(timestamps, u1, color='green', label='upper_1')
        plt.plot(timestamps, l2, color='red', label='lower_2')
        plt.plot(timestamps, u2, color='red', label='upper_2')
        plt.plot(timestamps, spread, color='black', label='spread')
        plt.legend(loc=0)
        plt.grid()
        plt.show()
        
    def plot_strategy(self):
        plot_str = 'Cumulative PNL daily ' + str(self.ric_long) + ' / ' + str(self.ric_short) + '\n'\
            + 'PNL annual mean ' + str(self.pnl_mean) + '\n'\
            + 'PNL annual volatility ' + str(self.pnl_volatility) + '\n'\
            + 'PNL annual Sharpe ' + str(self.pnl_sharpe)
        plt.figure()
        plt.title(plot_str)
        plt.xlabel('Time')
        plt.ylabel('Cum PNL')
        plt.plot(self.dataframe_strategy['date'], self.dataframe_strategy['cum_pnl_daily'], color='blue', label='mean')
        plt.grid()
        plt.show()
        
        








