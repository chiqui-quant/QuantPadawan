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
    "data_type" : "simulation", # simulation real custom
    "variable_name" : "student", # normal student chi-square SPY
    "degrees_freedom" : 9, # only in student and chi-square
    "ns_sims" : 10**6 # for simulated random variables
} # this is a dictionary with the input variables

dm = file_classes.distribution_manager(inputs)
dm.load_timeseries() # polymorphism
dm.compute()