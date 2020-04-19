# Basic Libraries for Quantitative Finance
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader.data as web
from matplotlib.finance import candlestick_ohlc
import matplotlib.dates as mdates
style.use(ggplot)

# Initalize basic variable. 
start = dt.datetime(2009, 1, 1)
end = dt.datetime(2019, 12, 31)

# Initalize a dataframe from a specific source during start and end. 
df = web.DataReader('TSLA', 'yahoo', start, end)
print(df.head())

# I. BASIC DATAFRAME MANIPULATION

# Convert the dataframe to a csv file, then use pandas to operates on the local csv file. 
df.to_csv('tsla')
df = pd.read_csv('tsla.csv', parse_dates = True, index_col = 0)

# Or we can choose to plot only one variable.
df['Adj Close'].plot()

# Creating data columns through calculation. 
df['100ma'] = df['Adj Close'].rolling(window=100, min_periods=0).mean()
df.dropna(inplace=True)
# inplace = true allows us to manipulate the df variable directly. 
print(df.head())


# II. BASIC PLOTTING.

df.plot()
plt.show()

ax1 = plt.subplot2grid((6,1), (0,0), rowspan = 5, colspan = 1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan = 1, colspan = 1, sharex=ax1)

ax1.plot(df.index, df['Adj Close'])
ax1.plot(df.index, df['100ma'])
ax2.bar(df.index, df['Volume'])

plt.show()

# III. Resampling

# Make a new dataframe through resample, get average per 10 days period.
# ohlc = 'Open High Low Close', i.e, market open price, highest price, lowest price, close price of the stock concerned.

df_olhc = df['Adj Close'].resample('10D').mean().ohlc()
df_volume = df['Volume'].resample('10D').sum()

# We need to reset the index so that date is columns. 
df_ohlc.reset_index(inplace=True)
df_ohlc['Date'] = df_ohlc['Date'].map(mdates.datet2num)

ax3 = plt.subplot2grid((6,1), (0,0), rowspan = 5, colspan = 1)
ax4 = plt.subplot2grid((6,1), (5,0), rowspan = 1, colspan = 1, sharex=ax3)
ax1.xaxis_date()

candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup = 'g')
ax4.fill_between(df_volume.index,map(mdates.date2num, df_volume.values, 0))
plt.show()

print(df_ohlc.head())
