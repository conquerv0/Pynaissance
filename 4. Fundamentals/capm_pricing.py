# Basic Setup
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels import regression
import matplotlib.pyplot as plt

# As usual, we can start by performing regression on a target asset.

start_date = '2018-01-01'
end_date = '2019-12-31'

R = get_pricing('MSFT', fields='price', start_date=start_date, end_date=end_date).pct_change()[1;]

# To obtain the risk-free proxy

R_F = get_pricing('BIL', fields='price',  start_date=start_date, end_date=end_date).pct_change()[1;]

# Calculate the beta against the market

M = get_pricing('SPY', start_date=start_date, end_date=end_date, fields='price').pct_change()[1:]

MSFT_results = regression.linear_model.OLS(R-R_F, sm.add_constant(M)).fit()
MSFT_beta = MSFT_results.params[1]

M.plot()
R.plot()
R_f.plot()
plt.xlabel('Time')
plt.xlabel('Daily Percent Return')
plt.legend()

MSFT_results.summary()

# After above summary is obtained, we can use the calculated beta to make predictions of the return using CAPM.

predictions = R_F+ MSFT_beta*(M - R_F) 

predictions.plot()
R.plot(color='R')
plt.legend(['Prediction', 'Actual Return'])

plt.xlabel('Time')
plt.ylabel('Daily Percent Return')






