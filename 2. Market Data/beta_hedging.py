# Basic Setup: Import Libraries
import numpy as np
from statsmodels import regression
import statsmodels.api as sm
import matplotlib.pyplot as plt
import math

start = '2018-01-01'
end = '2019-01-01'
asset = get_pricing('AMD', fields='price', start_date=start, end_date=end)
benchmark = get_pricing('SPY', fields='price', start_date=start, end_date=end)
