## Chapter3: Main Modules of Pytorch

> Motivation: 

### 1. Configuration

```Python
# Basic Imports
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optimizer
```

由于深度学习任务大致的趋同性，如下几个超参数一般作为`global`在任务开始部分设置:

- batch size
- learning rate(initial alpha)
- training epochs (max_epochs)

```Python
batch_size = 32
lr = 1e-4
max_epochs = 100
```



- Device Agnostic(GPU, CPU configuration)

```Python
# 方案一：使用os.environ，这种情况如果使用GPU不需要设置
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# 方案二：使用“device”，后续对要使用GPU的变量用.to(device)即可
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
```



### 2. Data Reader

Pytorch的数据读入通过Dataset, Dataloader的组合形式完成，在Dataset定义好数据格式和数据变换形式后，Dataloader通过自定义的iterator分批读入数据。

如以下例子:

class MyDataset(Dataset):
    def __init__(self, data_dir, info_csv, image_list, transform=None):
        """
        Args:
            data_dir: path to image directory.
            info_csv: path to the csv file containing image indexes
                with corresponding labels.
            image_list: path to the txt file contains image names to training/validation set
            transform: optional transform to be applied on a sample.
        """
        label_info = pd.read_csv(info_csv)
        image_file = open(image_list).readlines()
        self.data_dir = data_dir
        self.image_file = image_file
        self.label_info = label_info
        self.transform = transform

```Python
def __getitem__(self, index):
    """
    Args:
        index: the index of item
    Returns:
        image and its labels
    """
    image_name = self.image_file[index].strip('\n')
    raw_label = self.label_info.loc[self.label_info['Image_index'] == image_name]
    label = raw_label.iloc[:,0]
    image_name = os.path.join(self.data_dir, image_name)
    image = Image.open(image_name).convert('RGB')
    if self.transform is not None:
        image = self.transform(image)
    return image, label

def __len__(self):
    return len(self.image_file)
```

在构件好自定义的dataset后，就可以使用dataloader来分批读入数据：

```python
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, num_workers=4, shuffle=False)
```

### 3. Model Construction

#### 3.1 Neural Network

```python
import torch
from torch import nn

class MLP(nn.Module):
  # 声明带有模型参数的层，这里声明了两个全连接层
  def __init__(self, **kwargs):
    # 调用MLP父类Block的构造函数来进行必要的初始化。这样在构造实例时还可以指定其他函数
    super(MLP, self).__init__(**kwargs)
    self.hidden = nn.Linear(784, 256)
    self.act = nn.ReLU()
    self.output = nn.Linear(256,10)
    
   # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
  def forward(self, x):
    o = self.act(self.hidden(x))
    return self.output(o)   

```

#### 3.2 Common Network Layer

> 

#### 	1. Dense Layer(包含模型参数层)

#### 	2. 2D Convolutional Layer(二维卷积层)

#### 	3. Pooling Layer(池化层)



### 3.3 SOTA Model

	#### 1. LeNet

![3.4.1](https://github.com/datawhalechina/thorough-pytorch/raw/main/%E7%AC%AC%E4%B8%89%E7%AB%A0%20PyTorch%E7%9A%84%E4%B8%BB%E8%A6%81%E7%BB%84%E6%88%90%E6%A8%A1%E5%9D%97/figures/3.4.1.png)



```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 输入图像channel：1；输出channel：6；5x5卷积核
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 2x2 Max pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 如果是方阵,则可以只使用一个数字进行定义
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 除去批处理维度的其他所有维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)
```

```python
Net(
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
```



#### 2. AlexNet

![3.4.2](https://github.com/datawhalechina/thorough-pytorch/raw/main/%E7%AC%AC%E4%B8%89%E7%AB%A0%20PyTorch%E7%9A%84%E4%B8%BB%E8%A6%81%E7%BB%84%E6%88%90%E6%A8%A1%E5%9D%97/figures/3.4.2.png)

```python
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4), # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(3, 2), # kernel_size, stride
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
         # 这里全连接层的输出个数比LeNet中的大数倍。使用丢弃层来缓解过拟合
        self.fc = nn.Sequential(
            nn.Linear(256*5*5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
            nn.Linear(4096, 10),
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output

```

```python
net = AlexNet()
print(net)
```

```python
AlexNet(
  (conv): Sequential(
    (0): Conv2d(1, 96, kernel_size=(11, 11), stride=(4, 4))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(96, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (4): ReLU()
    (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU()
    (8): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU()
    (10): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU()
    (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (fc): Sequential(
    (0): Linear(in_features=6400, out_features=4096, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.5)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU()
    (5): Dropout(p=0.5)
    (6): Linear(in_features=4096, out_features=10, bias=True)
  )
)
```



### 4. Loss Function

	#### 	1. Binary Cross Entropy Loss 

```latex
$$ \ell(x, y)=\left{\begin{array}{ll} \operatorname{mean}(L), & \text { if reduction }=\text { 'mean' } \ \operatorname{sum}(L), & \text { if reduction }=\text { 'sum' } \end{array}\right. $$

```

#### 	2. Cross Entropy Loss

#### 	3. L1 Loss

#### 	4. MSE Loss

#### 	5. Possion Negative Logloss

### 5. Optimizer

### 6. Training and Evaluation

### 7. Visualization



