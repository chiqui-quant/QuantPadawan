from cmath import sqrt
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
            # directory = 'C:\\Users\\Chiqui\\Desktop\\Python Projects\\QuantPadawan\\data\\'

            # ric = self.inputs.variable_name
            # path = directory + ric + '.csv'
            # raw_data = pd.read_csv(path)
            # t = pd.DataFrame() # t (for table) with no elements
            # t['date'] = pd.to_datetime(raw_data['Date'], dayfirst=True) # if dayfirst is omitted we have American dates
            # t['close'] = raw_data['Close']
            # t.sort_values(by='date',ascending=True)
            # t['close_previous'] = t['close'].shift(1) # shift moves values below by the amount you indicate
            # t['return_close'] = t['close']/t['close_previous'] - 1
            # t = t.dropna()
            # t = t.reset_index(drop=True)
            
            # self.data_table = t
            # self.description = 'market data ' + ric
            # self.nb_rows = t.shape[0]
            # self.vec_returns = t['return_close'].values

            ric = self.inputs.variable_name
            t = file_functions.load_timeseries(ric)

    def plot_histogram(self):
        plt.figure()
        plt.hist(self.vec_returns,bins=100) # bins is the number of segments
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
        self.r_value = None
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
            + 'r-value (corr) ' + str(self.r_value)\
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
        self.r_value = np.round(r_value, self.nb_decimals) # correlation coefficient
        self.r_squared = np.round(r_value**2, self.nb_decimals) #pct of variance of y explained by x
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
        self.benchmark = inputs.benchmark # the market in CAPM (in general ^STOXX50E)
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
        beta_portfolio_usd = beta_portfolio * delta_portfolio # mln USD
        # Print input
        print('------')
        print('Input portfolio:')
        print('Delta mlnUSD for ' + security + ' is ' + str(delta_portfolio))
        print('Beta for ' + security + ' vs ' + benchmark + ' is ' + str(beta_portfolio))
        print('Beta mlnUSD for ' + security + ' vs ' + benchmark + ' is ' +  str(beta_portfolio_usd))
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

    def compute(self):
        # Exact solution using matrix algebra
        shape = [len(self.hedge_securities)]
        betas = self.betas
        deltas = np.ones(shape) # vertical vector of ones
        targets = -np.array([[self.delta_portfolio],[self.beta_portfolio_usd]]) # our targets in order to hedge are -delta and -beta
        mtx = np.transpose(np.column_stack((deltas,betas))) # stack deltas and betas and take the transpose
        self.optimal_hedge = np.linalg.inv(mtx).dot(targets) # invert the matrix and multiply by targets
        self.hedge_delta = np.sum(self.optimal_hedge)
        self.hedge_beta_usd = np.transpose(betas).dot(self.optimal_hedge).item()

        # Print result
        print('------')
        print('Optimization result')
        print('------')
        print('Delta: ' + str(self.delta_portfolio))
        print('Beta USD: ' + str(self.beta_portfolio_usd))
        print('------')
        print('Hedge delta: ' + str(self.hedge_delta))
        print('Hedge beta: ' + str(self.hedge_beta_usd))
        print('------')
        print('Betas for the hedge: ')
        print(betas)
        print('------')
        print('Optimal hedge: ')
        print(self.optimal_hedge)
        print('------')


class hedge_input():

    def __init__(self):
        self.benchmark = None # the market in CAPM (in general ^STOXX50E)
        self.security = 'BBVA.MC' # portfolio to hedge
        self.hedge_securities = ['STOXX50E', '^FCHI'] # hedge universe
        self.delta_portfolio = 10 # mlnUSD (default 10)


            






