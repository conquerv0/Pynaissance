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
      
 get_yahoo_data()
 
def compile_data():
  "This function parse csv files into pandas dataframes. "
  with open('sp500tickers.pickle', 'rb') as f:
    tickers = pcikle.load(f)
   
  main_df = pd.DataFrame()
  
  for count, ticker in enumerate(tickers):
    df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
    df.set_index('Date', inplace = True)
    
    df.rename(columns = {'Adj Close', ticker}, inplace=True)
    df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)
    
    if main_df.empty:
      main_df = df
    else:
      main_df = main_df.join(df, how='outer')
    
    if count % 10 == 0:
      print(count)
    
  print(main_df.head())
  main_df.to_csv('sp500_closes.csv')
 
#II. Data Visualization with correlation heat map.

def visualize_data():
  """"""
  df = pd.read_csv('sp500_joined_closes.csv')
  df_corr = df.corr()
  
  data = df_corr.values
  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1) # An one by one plot.
  
  heatmap = ax.pcolor(data, cmap = plt.cm.RdYlGn)
  fig.colorbar(heatmap)
  ax.set_xticks(np.arange(data.shape[0])+ 0.5, minor=False)
  ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)
  ax.inverst_yaxis()
  ax.xaxis.tick_top()
  
  column_labels = df_corr.columns
  row_labels = df_corr.index
  
  ax.set_xticklables(column_labels)
  ax.set_yticklabels(row_labels)
  plt.xticks(rotation=90)
  heatmap.set_clim(-1, 1)
  plt.tight_layout()
  plt.show()
  
visualize_data()
  
