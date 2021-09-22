import pytorch as torch
import numpy as np
import matplotlib as plt
import torch.nn as nn
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.metrics import mean_squared_error

global WIDTH
WIDTH = 50

class IdentityBlock(nn.Module):
  def __init__(self, width = WIDTH):
   
