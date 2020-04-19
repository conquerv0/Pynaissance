# Basic Imports
import bs4 as bs
import pickle
import request

# Some information about beautiful soup: To be updated. 
def sp500_tickers():
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
