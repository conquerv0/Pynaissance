# Basics Setup

import pandas as pd
import numpy as np
import talib
import sp500
import pandas.datareader as pd.datareader

"""
MACD Formula Components:
Short term EMA(Exponnential moving average, a type of moving average that assigns greater weights to most recent data point)
Long term EMA: Long term exponential moving average of index price. 
DIF: Difference between short term EMA and long term EMA. 
DEA: Difference Exponential Average, The moving average of the DIF line in the given market window.
MACD: The difference between DIF and DEA.

The basic idea of a MACD strategy:
Buy when DIF cross DIA from below to above.
Sell when DIF cross DEA from above to below"""

sp500.get_yahoo_data()
universe = sp500.compile_data()

short_window = 12
long_window

def handle_data(account): 
    
    all_close = account.get_attribute_history('closePrice', longest_history)
    long = []
    short = []
    
    for stk in universe[stock]:
        price = all_close[stk]
        if price is None:
            continue
        try:
            # Use ta-lib to calculate MACD
            macd_tmp = talib.MACD(prices, fastperiod=short_window, slowperiod=long_window)
            DIF = macd_tmp[0]
            DEA = macd_tmp[1]
            MACD = macd_tmp[2]
        except:
            continue
        
        # Determine what signal MACD provides
        if MACD[-1] > 0 and MACD[-4] < 0:
            long.append(stk)
        elif MACD[-1] < 0 and MACD[-4] > 0:
            short.append(stk)
            
        # Rebalance the current holding.
        
