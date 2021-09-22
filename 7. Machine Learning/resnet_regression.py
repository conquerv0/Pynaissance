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
      super().__init__()
      self.act = nn.Tanh()
      self.width = WIDTH
      self.layer1 = nn.Linear(WIDTH, WIDTH)
      self.layer2 = nn.Linear(WIDTH, WIDTH)
      self.batch_norm = nn.BatchNorm1d(WIDTH)
      
  def forward(self, x):
      x = self.act(self.batch_norm(self.layer1(x)))
      x = self.act(self.batch_norm(self.layer2(x) + x))
      return x
    

class Block(nn.Module):
  def __init__(self, input_dim, width=WIDTH):
      super().__init__()
      self.act = nn.Tanh()
      self.input_dim = input_dim
      self.layer1 = nn.Linear(input_dim, WIDTH)
      self.layer2 = nn.Linear(WIDTH, WIDTH)
      self.batch_norm = nn.BatchNorm1d(WIDTH)
      
 def forward(self, x):
      x = self.act(self.batch_norm(self.layer1(x)))
      x = self.act(self.batch_norm(self.layer2(x)))
      shortcut = self.act(self.batch_norm(self.layer1(x)))
      x = self.act(x + shortcut)
      return x
    
 
   
