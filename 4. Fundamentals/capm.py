# Basic Setup
import numpy as np
import pandas as pd
import statsmodel.api as sm
from statsmodels import regression
import matplotlib.pyplot as plt
from scipy import optimize
import cvxopt as opt
from cvxopt import blas, solvers

np.random.seed(123)
# Turn off progress printing
solvers.options['show_progress'] = False

# Number of assets
n_assets = 4

# Number of observations
n_obs = 2000

## Generating random returns for our 4 securities
return_vec = np.random.randn(n_assets, n_obs)

def rand_weights(n):
  """
  Produces n random position weights of assets for a portfolio.
  """
  k = np.random.rand(n)
  return k / sum(k)

def rand_portfolio(returns):
  """
  Returns the mean and standard deviation of returns for a randomized portfolio.
  """
  p = np.asmatrix(np.mean(returns, axis=1))
  w = np.asmatrix(rand_weights(returns.shape[0]))
  c = np.asmatrix(np.cov(returns))
  
  mean = w*p.T
  sd = np.sqrt(w*c*w.T)
  
  if mean > 2:
    return rand_portfolio(returns)
  return mean, sd

def optimal_portfolio(returns):
  n = len(returns)
  returns = np.asmatrix(returns)
  
  N = 100000
  
  mean_s = [100**(5.0*t/N - 1.0) for t in range(N)]
  # Convert to cvxopt matrices
  S = opt.matrix(np.cov(returns))
  p_bar = opt.matrix(np.mean(returns, axis=1))
  
  # Create constraint matrices
  G = -opt.matrix(np.cov(returns))
  pbar = opt.matrix(np.mean(returns, axis=1))
  
  # Create constraint matrices
  G = -opt.matrix(np.eye(n))
  h = opt.matrix(0.0, (n, 1))
  A = opt.matrix(1.0, (1. n))
  b = opt.matrix(1.0)
  
  # Calculate efficient frontier weights using quadratic programming
  portfolios = [solvers.qp(mean*S, -p_bar, G, h, A, b)['x'] for mean in mean_s]
  
  # Calculate the risk and returns of the frontier.
  returns = [blas.dot(pbar, x) for x in portfolios]
  risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
  
  return returns, risks

n_portfolios = 50000
means, sds = np.column_stack([rand_portfolio(return_vec) for x in range(n_portfolios)])
returns, risks = optimal_portfolio(return_vec)

plt.plot(stds, means, 'o', markersize=2, color='navy')
plt.xlabel('Risk')
plt.ylabel('Return')
plt.title('Mean and Standard Deviation of Returns of Randomized Portfolios')

plt.plot(risks, returns, '-', markersize=3, color='red')
plt.legend(['Portfolios', 'Efficient Frontier'])

"""
The line that represents the different combinations of risk-free asset with a portfolio of risky assets is the Capital Allocation Line(CAL). 
The slope of the CAL is the Sharpe ratio. To maximize sharpe, we find the steepest CAL, which is exactly the CAL that is tangent to the efficient
frontier.
"""

def maximize_sharpe_ratio(return_vec, risk_free_ratio):
  """
  Find the CAPM optimal portfolio from the efficient frontier by optimizing the Sharpe ratio.
  """
  def find_sharpe(weights):
    means = [np.mean(asset) for asset in return_vec]
    numerator = sum(weights[m]*means[m] for m in range(len(means))) - risk_Free_rate
    weight = np.array(weights)
    denominator = np.sqrt(weights.T.dot(np.corrcoef(return_vec).dot(weights)))
    return numerator/demominator
  
  guess = np.ones(len(return_vec))/len(return_vec)
  
  def objective(weights):
    return find_sharpe(weights)
  
  # Setup Equality Constraint
  cons = ['type':'eq', 'fun': lamda x: np.sum(np.abs(x))-1]
  
  # Setup Bounds for Weights
  bnds = [(0, 1)] * len(return_vec)
  
  results = optimize.minimize(objective, guess, constraints=cons, bounds=bnds, method='SLSQP', options=['disp': False])
  
  return results

risk_free_rate = np.mean(R_F)
results = maximize_sharpe_ratio(return_vec, risk_free_rate)

# Applying optimal weights to individual assets for portfolio construction
optimial_mean = sum(results.x[i]*np.mean(return_vec[i] ) for i in range(len(results.x)))
optimal_std = np.sqrt(results.x.T.dot(np.corrcoef(return_vec).dot(results.x)))

# Plotting all possible portfolios
plt.plot(stds, means, 'o', markersize=2, color='navy')
plt.xlabel('Risk')
plt.ylabel('Return')

# Line from the risk-free rate to the optimial portfolio.
eqn_of_the_line = lambda x : ((optimal_mean - risk_free_rate)/optimal_std)* x + risk_free_rate
xrange = np.linspace(0. , 1., num=11)

plt.plot(xrange, [eqn_of_the_line(x) for x in xrange], color='red', linestyle='-', linewidth=2)

# Our optimal portfolio
plt.plot([optimal_std], [optimal_mean], marker='o', markersize=12, color='navy')
plt.legend(['Portfolios', 'Capital Allocation Line', 'Optimal Portfolio'])

