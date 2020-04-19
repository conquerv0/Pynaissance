# Basic Libraries for Quantitative Finance
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader.data as web
from matplotlib.finance import candlestick_ohlc
import matplotlib.dates as mdates
style.use(ggplot)

# 1. Basic setup

# Initalize basic variable. 
start = dt.datetime(2009, 1, 1)
end = dt.datetime(2019, 12, 31)

# Initalize a dataframe from a specific source during start and end. 
df = web.DataReader('GOOGL', 'yahoo', start, end)
print(df.head())

# Convert the dataframe to a csv file, then use pandas to operates on the local csv file. 
df.to_csv('googl')
df = pd.read_csv('googl.csv', parse_dates = True, index_col = 0)

# II. Resampling 

# Make a new dataframe through resample, get average per 10 days period.
# ohlc = 'Open High Low Close', i.e, market open price, highest price, lowest price, close price of the stock concerned.

df_olhc = df['Adj Close'].resample('10D').mean().ohlc()
df_volume = df['Volume'].resample('10D').sum()

# We need to reset the index so that date is columns. 
df_ohlc.reset_index(inplace=True)
df_ohlc['Date'] = df_ohlc['Date'].map(mdates.datet2num)

# II. Candlestick graph

ax3 = plt.subplot2grid((6,1), (0,0), rowspan = 5, colspan = 1)
ax4 = plt.subplot2grid((6,1), (5,0), rowspan = 1, colspan = 1, sharex=ax3)
ax1.xaxis_date()

candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup = 'g')
ax4.fill_between(df_volume.index,map(mdates.date2num, df_volume.values, 0))
plt.show()

print(df_ohlc.head())
