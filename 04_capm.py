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
security = 'BBVA.MC' # variable y

# Load synchronized timeseries
t = file_functions.load_synchronized_timeseries(ric_x=benchmark, ric_y=security)

