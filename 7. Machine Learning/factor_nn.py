import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

""" == Data Preproc Module =="""

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

""" == Training Module == """

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
         self.layers = nn.Sequential(
            nn.Linear(416, 100),
            nn.Tanh(),
            nn.Linear(100, 10),
            nn.Tanh(),
            nn.Linear(10, 1)
        )
    def forward(self, x):
        x = self.layers(x)
        return x
    
def train(model, device, dataloader, epochs=200)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_func = nn.MSELoss()
    mse, corrs = [], []
    
    print('Start training model...')
    
    for epoch in range(epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = loss_func(y_hat, y)
            
            if device == torch.device('cuda'):
                y_hat = y_hat.cpu().data.numpy()
                y = y.cpu().data.numpy()
            corr = np.corrcoef(y_hat.flatten(), y.flatten())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        mse.append(loss)
        corrs.append(corr)
        
        if epoch % 10 == 0:
            print('Step: {}, Loss: {}, Correlation: {}'.format(epoch, loss.item(), corrs[-1]))
        if epoch % 20 == 0:
            plt.plot(range(len(corrs)), corrs, label='correlation')
            plt.show()
        
    fig, axs = plt.subplots(2, sharex=True, tightlayout=True)
    axs[0].plot(range(epochs), corrs, label='correlation')
    axs[1].plot(range(epochs), mse, color='salmon', label='mse')
    plt.title('Change in Correlation and MSE')
    
    return model

if __name__=="__main__":
  pass
