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

  

