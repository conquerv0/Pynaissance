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

def initialize(account):
    pass

def handle_data(account):    
    #print account.current_date
    cal = Calendar('China.SSE')
    last_dayt1 = cal.advanceDate(account.current_date, str(-t1)+'B', BizDayConvention.Preceding).toDateTime()           #计算出前t1个交易日
    buylist = []
    selllist = []

