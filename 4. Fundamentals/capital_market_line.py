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
