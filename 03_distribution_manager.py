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
# (1) Using a dictionary (but if it is removed it can be problematic to see which inputs are required)
# inputs = {
#     "data_type" : "real", # simulation real custom
#     "variable_name" : "A2A.MI", # normal student chi-square exponential BBVA.MC
#     "degrees_freedom" : None, # only in student and chi-square
#     "nb_sims" : None, # for simulated random variables
# } # this is a dictionary with the input variables

# (2) Using a dataframe 
# inputs_df = pd.DataFrame()
# inputs_df['data_type'] = ['real']
# inputs_df['variable_name'] = ['A2A.MI']
# inputs_df['degrees_freedom'] = [9]
# inputs_df['nb_sims'] = [10**6]
# Note: in this case since it is a dataframe you have to add[0] in the class (eg. self.inputs['variable_name'][0])

# (3) Calling single variables (but if you add or change something you have to modify attribute and class)
# input_data_type = "real" # simulation real custom
# input_variable_name = "A2A.MI" # normal student chi-square exponential BBVA.MC
# input_degrees_freedom = None # only in student and chi-square
# input_nb_sims =  None # for simulated random variables

# (4) Using a class: this is more 'robust' because we define inputs_class and make changes without breaking the rest of the code
inputs = file_classes.distribution_input()
inputs.data_type = 'real'
inputs.variable_name = 'A2A.MI'
inputs.degrees_freedom = None
inputs.nb_sims = None

# Note: using the functionality of "Go to declaration" it is possible to see where the variables are defined (this is why the 4th approach might be preferred).
# E.g. going to the declaration for distribution_manager brings us to where we defined it in file_classes, this way we can immediately see what is inside the black-box and add new extensions of the code there while keeping this code clear and functional.
dm = file_classes.distribution_manager(inputs) # initialize constructor
dm.load_timeseries() # polymorphism: get the timeseries
dm.compute() # compute returns and all different risk metrics
dm.plot_histogram() # plot histogram
print(dm) # write all data in the console