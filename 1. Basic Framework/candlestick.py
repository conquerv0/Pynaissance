# Basic Import Libaries

# I. Resampling 

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
