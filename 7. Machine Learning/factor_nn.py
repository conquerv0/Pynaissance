import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

def data_parser(filepath):
    dataset = {}
    
    # read data from a matlab files. 
    try: 
      with h5py.File(filepath, 'r') as f:
        for k, v in f.items():
          dataset[k] = np.array(v)
    except:
      import scipy.io
      mat = scipy.io.loadmat(filepath)
      dataset['x'] = mat['x'].T
      dataset['Y'] = mat['y'].T
    
    print('dataset is loaded into numpy array.')
    return dataset

def data_preproc(dataset):
  pass

def net(nn.Module):
  def __init__(self):
    super(net, self).__init__()
    
def train:
  pass

if __name__=="__main__":
  pass
