# Basic Setup
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
