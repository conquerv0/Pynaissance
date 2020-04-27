# Basic Import Library
import numpy as np
import pandas as pd
import pickle

def process_data_for_labels(ticker):
  days = 7 # the training time period at which we are interested in rebalancing
  df = pd.read_csv('sp500.closes.csv' index_col=0)
  tickers = df.columns.values.tolist()
  df.fillna(0, inplace=True)
  
  for i in range(1, days+1):
    # we need to shift the future data to the upper (older) column
    df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]
    
  df.fillna(0, inplace=True)
  return tickers, df

process_data_for_labels('XOM')
    
