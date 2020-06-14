# Basic Setup
from datetime import *
import time
import pandas as pd
import numpy as np
import math

start = datetime(2015, 1, 1)
end = datetime(2019, 5, 31)
benchmark = 'SPY'
capital_base  = 100000

t1 = 50 # Length of Moving Average window.
t2 = 30 # Length of Rate of Change(ROC) window.
MaxBar = 0.8 # When reach max holding period, selling point. 
