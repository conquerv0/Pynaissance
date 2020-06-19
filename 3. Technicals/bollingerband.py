# Basic Imports
import bs4 as bs
import matplot.pyplot as plt
from matplotlib import style
import numpy as np
import pickle
import request
import datetime as dt
import os
import pandas as pd
import pandas_datareader.data as web
from S&P500 import sp500_tickers, get_yahoo_data

style.use('ggplot')

def timestamp_to_datetime(timestamp):
    return datetime.fromtimestamp(float(timestamp))

df = pd.read_csv(‘’)
df = df['Price'].resample('D').ohlc()


# To Calculate Bollinger Bands
window = 30
n_std = 1.5 # number of Standard Deviation.

rolling_u = df['close'].rolling(window).mean() # rolling mean
rolling_std = df['close'].rolling(window).std() 

df['Rolling Mean']=rolling_u
df['Bollinger High'] = rolling_u + (rolling_std*n_std)

df[['close', 'Bollinger High', 'Bollinger Low']].plot()


# Strategy Implementation.

df['Position']= None # define a new column for the asset positions.

# Set asset positons through the following binary rule. Note this involves 
# the idea of pair trading, which we will discuss later.

open = True
for i in range(len(df)):
  row = df.iloc[i]
  last_row = df.iloc[i-1]
  
  if open and row['close'] < row['Bollinger Low'] and last_row['close'] > last_row['Bollinger Low']:
    df.iloc[i, df.columns.get_loc('Position')] = 1
    open = False
   
  if not open and row['close'] > row['Bollinger High'] and last_row['close'] < last_row['Bollinger High']:
    df.iloc[i, df.columns.get_loc('Position')] = -1
  
  df.dropna(subset=['Position'])

  
 # Visualize strategy changes.
df[['close', 'Rolling Mean', 'Bollinger High', 'Bollinger Low']].plot(figsize=(8, 6))
for i, pos in df.dropna(subset=['Position'])['Position'].iteritems():
  plt.axvline(i, color='green' if pos == 1 else 'red')
  
# Visualize and Evaluate Strategy Return
df.['Position'].fillna(method='ffill', inplace=True)

# Calculate daily return
df['Market Return'] = df['close'].pct_change
df['Strategy Return'] = df['Market Return'] * df['Positions']

df['Strategy Return'].cumsum().plot(figsize=(8, 6))


