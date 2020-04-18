# Basic Libraries for Quantitative Finance

import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader.data as web

style.use(ggplot)

# Initalize basic variable. 
start = dt.datetime(2018, 1, 1)
end = dt.datetime(2019, 12, 31)

# Initalize a dataframe from a specific source during start and end. 
df = web.DataReader('TSLA', 'yahoo', start, end)
print(df.head())
