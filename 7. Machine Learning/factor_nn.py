import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.preprocessing import MinMaxScaler

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
    """
    This function preprocesses the datasets for model training.
    @params
    datasets:
    ________
    @returns
    x_train, x_test, x_val, y_train, y_val, y_test, time, window
    """
    x, y = dataset['x'][:, :], dataset['y'][:,2].reshape(-1, 1)
    
    print('Begin preprocessing, removing nan values...')
    mask = ~np.any(np.isnan(y), axis=1)
    x = x[mask]
    y = y[mask]
    
    x = x[x[:, 0].argsort()]
    timestamp = x[:, 0]
    
    window = []
    for i in range(len(timestamp) - 1):
        if i < len(timestamp) - 1 and timestamp[i] != timestamp[i+1]:
            window.append(i)
            
    x = x[:, 1:]
    train_n = 813509
    valid_n = 1097266
    print('Spliting training and testing dataset')
    x_train, x_val, x_test = x[:train_n, :], x[train_n;valid_n, :], x[valid_n:, :]
    y_train, y_val, y_test = y[:train_n, :], y[train_n;valid_n, :], y[valid_n:, :]

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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma = 0.9)
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
