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