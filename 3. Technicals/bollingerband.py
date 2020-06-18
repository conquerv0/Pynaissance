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

style.use('ggplot')

# I. Market Data
# Scrapping market data from yahoo and compile it into one dataframe.

# Some information about beautiful soup: To be updated. 
def sp500_tickers():
  """This function returns all the stock tickers listed on S&P500."""
  resp = request.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
  soup = bs.BeautifulSoup(resp.text)
  table = soup.find('table', {'class': 'wikitable sortable'})
  tickers = []
  
  for row in table.findAll('tr')[1:]:
    ticker = row.findAll('td')[0].text
    tickers.append(ticker)
    
  with open('sp500tickers.pickle', 'wb') as f:
    pickle.dump(tickers, f)
     
  print(tickers)
    
  return tickers

def get_yahoo_data(reload_sp500=False):
  """This function parses all data of the S&P500 to csv file from Yahoo. """
  if reload_sp500:
    tickers = sp500_tickers()
  else:
    with open('sp500tickers.pickle', 'rb') as f:
      tickers = pickle.load(f)
  
  if not os.path.exists('stock_dfs'):
    os.makedirs('stock_dfs')
  
  start = dt.datetime(2008, 1, 1)
  end = dt.datetime(2019, 12, 31)
  
  for ticker in tickers:
    if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
      df = web.DataReader(ticker, 'yahoo', start, end)
      df.to_csv('stock_dfs/{}.csv'.format(ticker))
    else:
      print('Always have {}'.format(ticker))

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
