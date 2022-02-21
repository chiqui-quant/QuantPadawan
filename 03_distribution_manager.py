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

# There are 4 ways in which we can define inputs
# Using a dictionary (but if it is removed it can be problematic to see which inputs are required)
# inputs = {
#     "data_type" : "real", # simulation real custom
#     "variable_name" : "BBVA.MC", # normal student chi-square exponential BBVA.MC
#     "degrees_freedom" : None, # only in student and chi-square
#     "nb_sims" : None, # for simulated random variables
# } # this is a dictionary with the input variables

# Using a dataframe (most convenient)
# inputs_df = pd.DataFrame()
# inputs_df['data_type'] = ['real']
# inputs_df['variable_name'] = ['BBVA.MC']
# inputs_df['degrees_freedom'] = [9]
# inputs_df['nb_sims'] = [10**6]
# Note: in this case since it is a dataframe you have to add[0] in the class (eg. self.inputs['variable_name'][0])


# Calling single variables (but if you add or change something you have to modify attribute and class)
# input_data_type = "real" # simulation real custom
# input_variable_name = "BBVA.MC" # normal student chi-square exponential BBVA.MC
# input_degrees_freedom = None # only in student and chi-square
# input_nb_sims =  None # for simulated random variables

# This is more 'solid' because we have defined inputs_class and make changes without breaking the rest of the code
inputs_class = file_classes.distribution_input()
inputs_class.data_type = 'real'
inputs_class.variable_name = 'BBVA.MC'

dm = file_classes.distribution_manager(inputs_class) #initialize constructor
dm.load_timeseries() # polymorphism: get the timeseries
dm.compute() # compute returns and all different risk metrics
dm.plot_histogram() # plot histogram
print(dm) # write all data in the console