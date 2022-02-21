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


# Inputs
benchmark = '^STOXX50E' # variable x
security = 'BBVA.MC' #variable y

# Load 2 timeseries
table_x = file_functions.load_timeseries(benchmark)
table_y = file_functions.load_timeseries(security)
