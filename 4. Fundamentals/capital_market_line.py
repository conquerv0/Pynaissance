# Basic Setup
import numpy as np
import pandas as pd
import statsmodel.api as sm
from statsmodels import regression
import matplotlib.pyplot as plt
from scipy import optimize
import cvxopt as opt
from cvxopt import blas, solvers

risk_free_rate = np.mean(R_F)

# We have two coordinates that we can use to map the SML(Security Market Line):(0, risk-free rate) and (1, market return)

line_eqn = lambda x : ((np.mean(M) - risk_free_rate) / 1.0)*x + risk_free_rate
xrange = np.linspace(0., 2.5, num=2)
plt.plot(xrange, [line_eqn(x) for x in xrange], color='red', linestyle='-', linewidth=2)

plt.plot([1], [np.mean(M)], marker='o', color='navy', markersize=10)
plt.annotate('Market', xy=(1, np.mean(M)), xytest=(0.9, np.mean(M)+0.00004))

# At this point, we need to compare to see if stocks in more cyclical industries yield higher betas. 

# Non-Cyclical Industry Stocks
non_cyclical = ['PG', 'DUK', 'PFE']
non_cyclical_returns = get_pricing(non_cyclical, fields='price', start_date=start_date, end_date=end_date).pct_change()[1:]
non_cyclical_returns.columns = map(lambda x : x.symbol, non_cyclical_returns.columns)

non_cyclical_betas = [regression.linear_model.OLS(non_cyclical_returns[asset], sm.add_constant(M)).fit().params[1] for asset in non_cyclical]

for asset, beta in zip(cyclical, cyclical_betas):
  plt.plot([beta], [np.mean(cyclical_returns[asset])], marker='o', color='y', markersize=10)
  plt.annotate(asset, 
               xy=(asset, xy=(beta, np.mean(cyclical_returns[asset])), 
               xytext=(beta + 0.015, np.mean(cyclical_returns[asset]) + 0.00025))

plt.plot([cylical_betas[2], cyclical_betas[2]],
        [np.mean(cyclical_returns.iloc[:,2]), 
         line_eqn(cyclical_betas[2])],
        color='grey')
plt.annotate('Alpha', 
             xy=(cyclical_betas[2] + 0.05, 
                 (line_eqn(cyclical_betas[2])-np.mean(cyclical_returns.iloc[:, 2]))/2+np.mean(cylical_returns.iloc[:,2])),
             xytext=(cyclical_betas[2] + 0.05,
                 (line_eqn(cyclical_betas[2])-np.mean(cyclical_returns.iloc[:, 2]))/2+np.mean(cylical_returns.iloc[:,2]))
            )
plt.xlabel('Beta')
plt.ylabel('Return')

