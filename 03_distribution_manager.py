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

inputs = {
    "data_type" : "real", # simulation real custom
    "variable_name" : "BBVA.MC", # normal student chi-square exponential BBVA.MC
    "degrees_freedom" : None, # only in student and chi-square
    "nb_sims" : None, # for simulated random variables
} # this is a dictionary with the input variables

dm = file_classes.distribution_manager(inputs)
dm.load_timeseries() # polymorphism
dm.compute()
dm.plot_histogram()
print(dm)