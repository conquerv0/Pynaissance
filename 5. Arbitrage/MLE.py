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
