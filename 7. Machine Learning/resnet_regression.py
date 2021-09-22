import pytorch as torch
import numpy as np
import matplotlib as plt
import torch.nn as nn
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
      self.batch_norm = nn.BatchNorm1d(WIDTH)
      
  def forward(self, x):
      x = self.act(self.batch_norm(self.layer1(x)))
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
      x = self.act(self.layer1(x))
      x = self.act(self.layer2(x))
      shortcut = self.act(self.layer1(x))
      x = self.act(x + shortcut)
      return x
    
class ResNet(nn.Module);
  def __init__(self, input_dim, width=WIDTH, layer=2):
      super().__init__()
      self.width = width
      self.input_dim = input_dim
      self.layers = nn.ModuleList()
      self.depth = layer
      
      self.layers.append(nn.Linear(input_dim, width))
      for _ in range(self.depth):
        self.layers.append(IdentityBlock(width))
        self.layers.append(IdentityBlock(width))
        self.layers.append(Block(width, width))
        
      self.fc = nn.Linear(Block(width, width))
  
  def forward(self, x):
      for layer in self.layers:
        x = layer(x)
      x = self.fc(x)
      return x
      
   
