# Basic Imports
import bs4 as bs
import pickle
import request
import datetime as dt
import os
import pandas as pd
import pandas_datareader.data as web

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
  
