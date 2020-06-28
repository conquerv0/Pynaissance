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
