# Basic Setup
from datetime import *
import time
import pandas as pd
import numpy as np
import math

start = datetime(2015, 1, 1)
end = datetime(2019, 5, 31)
benchmark = 'SPY'
universe = set_universe('SP500')
capital_base  = 100000

t1 = 50 # Length of Moving Average window.
t2 = 30 # Length of Rate of Change(ROC) window.
MaxBar = 0.8 # When reach max holding period, selling point. 

T =  pd.Series(data=[t1],index = universe)
commission = Commission(buycost=0.0003, sellcost=0.0003) 

def initialize(account):
    pass

def handle_data(account):    
    cal = Calendar('NYSE') # Account for API difference
    last_dayt1 = cal.advanceDate(account.current_date, str(-t1)+'B', BizDayConvention.Preceding).toDateTime()           
    # Calculate t1 trading day. 
    buylist = []
    selllist = []

