import pandas as pd
import numpy as np
import matplotlib as mpl 
import scipy
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2

class distribution_manager():
    
    def __init__(self, inputs):
        self.inputs = inputs
        self.data_table = None
        self.description = None
        self.nb_rows = None

    def __str__(self):
        str_self = self.description + ' | size ' + str(self.nb_rows) + '\n' + self.plot_str()
        return str_self

    def load_timeseries(self):
        
        data_type = self.inputs['data_type']
        
        if data_type == 'simulation':
            nb_sims = self.inputs['nb_sims']
            dist_name = self.inputs['variable_name']
            degrees_freedom = self.inputs['degrees_freedom']
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

            ric = self.inputs['variable_name']
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
            self.vec_returns = t['return_close'].values

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

    def plot_str(self):
        nb_decimals = 4
        plot_str = 'mean ' + str(np.round(self.mean, nb_decimals))\
            + ' | std dev ' + str(np.round(self.std, nb_decimals))\
            + ' | skewness ' + str(np.round(self.skew, nb_decimals))\
            + ' | kurtosis ' + str(np.round(self.kurtosis, nb_decimals)) + '\n'\
            + 'Jarque Bera  ' + str(np.round(self.jb_stat, nb_decimals))\
            + ' | p-value ' + str(np.round(self.p_value, nb_decimals))\
            + ' | is normal ' + str(self.is_normal)
        return plot_str






