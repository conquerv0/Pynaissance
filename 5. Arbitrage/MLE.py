# This program was originally authored by Delaney Granizo and Andrei Kirilenko 
as a part of the Master of Finance curriculum at MIT Sloan. 

"""
In this notebook, we mainly explores the statistical method of computing maximum likelyhood function
for common distributions. A subsequent financial application will be examined through fitting the normal
distributions to asset returns using MLE.
"""

# Basic Imports
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.stats

"""
== Normal Distribution ==

We first explores some basic sampling from a normal distribution. Given the data available, a
function can be defined so that it will compute the MLE for the \mu and \sigma parameters of the
normal distribution.

"""

True_Mean = 80
True_Std = 10
X = np.random.normal(True_Mean, True_Std, 1000)

def normal_mu_MLE(X):
  """Return the MLE given mu in a normal distribution.
  """
  n_obs = len(X)
  sum_obs = sum(x)
  return 1.0/n_obs*sum_obs

def normal_sigma_MLE(X):
  """Return 
  """
  n_obs = len(X)
  mu = normal_mu_MLE(X)
  sum_sqd = sum(np.power((X - mu), 2)) # sum up the squared differences.
  sigma_sq = 1.0/ n_obs * sum_sqd
  return math.sqrt(sigma_sq)

pdf = scipy.stats.norm.pdf
x = np.linespace(0, 80, 80)
plt.hist(x, bins=x, normed='true')
plt.plot(pdf(x, loc=mu, scale=std))
plt.xlabel('Value')
plt.ylabel('Observed Frequency')
plt.legend(['Fitted Distribution PDF', 'Observed Data', ])
  
# Exponential Distribution
TRUE_LAMDA = 5
X = np.random.exponetial(TRUE_LAMDA, 1000)

def exp_lamda_MLE(X):
  T = len(X)
  s = sum(X)
  return s/T

pdf = scipy.stats.exon.pdf
x = range(0, 80)
plt.hist(X, bins=x, normed='true')
plt.plot(pdf(x, scale=1))
plt.xlabel('Value')
plt.ylabel('Observed Frequency')
plt.legend(['Fitted Distribution PDF', 'Observed Data', ])

prices = get_pricing('AAPL', fields='price', start_date='2018-01-01', end_date='2019-12-31')
absolute_returns = np.diff(prices)
returns = absolute_returns / prices[:-1]
